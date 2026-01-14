"""
GPT-OSS Inference Script for SLM2 Benchmarking

Runs inference on slm2_test.jsonl using GPT-OSS from OpenRouter.
Generates predictions in the same format as slm2_output.jsonl.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from openai import OpenAI


class GPTOSSInference:
    """Handles inference using GPT-OSS model from OpenRouter."""
    
    def __init__(self, api_key: str = None, model_name: str = "openai/gpt-oss-120b:free"):
        """
        Initialize GPT-OSS inference client.
        
        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
            model_name: Model identifier on OpenRouter
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        print(f"Initialized GPT-OSS client with model: {model_name}")
    
    def format_input_triples(self, input_triples: List[List[str]]) -> str:
        """
        Format input triples for the prompt.
        
        Args:
            input_triples: List of [subject, predicate, object] triples
            
        Returns:
            Formatted string representation of triples
        """
        formatted = []
        for triple in input_triples:
            formatted.append(f"[{triple[0]}, {triple[1]}, {triple[2]}]")
        return ", ".join(formatted)
    
    def construct_prompt(self, instruction: str, input_triples: List[List[str]]) -> str:
        """
        Construct the prompt for GPT-OSS.
        
        Args:
            instruction: Task instruction
            input_triples: Knowledge graph triples
            
        Returns:
            Complete prompt string
        """
        triples_str = self.format_input_triples(input_triples)
        
        prompt = f"""{instruction}

Knowledge Graph Facts:
{triples_str}

IMPORTANT RULES:
1. If any triple contains "SOME_VALUE" or "INTERMEDIATE" as a placeholder, you MUST respond with EXACTLY: "I don't know."
2. If all facts are complete (no placeholders), generate a natural language sentence that accurately expresses the facts.
3. Do not add information not present in the facts.
4. Do not use words like "probably", "might", or other uncertainty markers unless saying "I don't know."

Response:"""
        
        return prompt
    
    def get_prediction(self, instruction: str, input_triples: List[List[str]], 
                      use_reasoning: bool = False, max_retries: int = 5) -> str:
        """
        Get prediction from GPT-OSS model with retry logic.
        
        Args:
            instruction: Task instruction
            input_triples: Knowledge graph triples
            use_reasoning: Whether to enable reasoning mode
            max_retries: Maximum number of retries for rate limits
            
        Returns:
            Model's predicted output
        """
        import time
        
        prompt = self.construct_prompt(instruction, input_triples)
        
        for attempt in range(max_retries):
            try:
                # Prepare API call parameters
                api_params = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.0,  # Deterministic output
                    "max_tokens": 200,
                }
                
                # Add reasoning if enabled
                if use_reasoning:
                    api_params["extra_body"] = {"reasoning": {"enabled": True}}
                
                response = self.client.chat.completions.create(**api_params)
                
                # Extract content
                prediction = response.choices[0].message.content.strip()
                
                return prediction
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors with exponential backoff
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8, 16 seconds
                        print(f"\nRate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\nMax retries reached. Returning fallback.")
                        return "I don't know."
                
                # For other errors, return fallback immediately
                print(f"\nError during inference: {e}")
                return "I don't know."  # Fallback on error
        
        return "I don't know."
    
    def run_inference_on_dataset(self, test_file: str, output_file: str, 
                                 use_reasoning: bool = False, rate_limit_delay: float = 4.0):
        """
        Run inference on the entire test dataset with rate limiting.
        
        Args:
            test_file: Path to slm2_test.jsonl
            output_file: Path to save predictions
            use_reasoning: Whether to enable reasoning mode
            rate_limit_delay: Delay in seconds between requests (default 4s for ~15 req/min)
        """
        import time
        
        test_path = Path(test_file)
        output_path = Path(output_file)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # Load test data
        print(f"Loading test data from {test_path}...")
        test_examples = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_examples.append(json.loads(line))
        
        print(f"Loaded {len(test_examples)} test examples")
        
        if use_reasoning:
            print("⚡ Reasoning mode ENABLED")
        else:
            print("💬 Standard mode (no reasoning)")
        
        # Calculate estimated time
        estimated_time_min = (len(test_examples) * rate_limit_delay) / 60
        print(f"⏱️  Estimated time: {estimated_time_min:.1f} minutes (with {rate_limit_delay}s delay)")
        
        # Run inference
        results = []
        print("\nRunning inference...")
        
        for i, example in enumerate(tqdm(test_examples, desc="Processing")):
            prediction = self.get_prediction(
                example['instruction'],
                example['input'],
                use_reasoning=use_reasoning
            )
            
            result = {
                'id': i,
                'input': example['input'],
                'gold_output': example['output'],
                'predicted_output': prediction,
                'label': example['label'],
                'hop_count': example['hop_count']
            }
            
            results.append(result)
            
            # Add delay between requests to avoid rate limits
            if i < len(test_examples) - 1:  # Don't delay after last request
                time.sleep(rate_limit_delay)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✅ Inference complete! Saved {len(results)} predictions.")
        print(f"\nNext steps:")
        print(f"1. Run evaluation: python slm2_evaluation.py (modify to use {output_path.name})")
        print(f"2. Or modify slm2_evaluation.py to accept command-line arguments")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GPT-OSS inference on SLM2 test dataset")
    parser.add_argument(
        '--test-file',
        type=str,
        default='../data/test/slm2_test.jsonl',
        help='Path to test dataset'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='gpt_oss_output.jsonl',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-oss-120b:free',
        help='Model name on OpenRouter'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=4.0,
        help='Delay between requests in seconds (default: 4.0 for ~15 req/min)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)'
    )
    parser.add_argument(
        '--reasoning',
        action='store_true',
        help='Enable reasoning mode'
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = GPTOSSInference(
        api_key=args.api_key,
        model_name=args.model
    )
    
    # Run inference
    inference.run_inference_on_dataset(
        test_file=args.test_file,
        output_file=args.output_file,
        use_reasoning=args.reasoning,
        rate_limit_delay=args.delay
    )


if __name__ == '__main__':
    main()
