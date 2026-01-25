"""
Inference script for Llama-3.2-3B (Vanilla - No Fine-tuning)
Tests fact verification on llm_test.jsonl dataset
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
from tqdm import tqdm

def load_model(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda"):
    """Load Llama-3.2-3B model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Model loaded successfully!")
    return model, tokenizer

def verify_claim(model, tokenizer, claim, max_length=128):
    """Verify if a claim is true or false"""
    
    # Format using Llama 3.2 chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant that verifies factual claims. Answer only with 'True' or 'False'."},
        {"role": "user", "content": f"Is the following claim factually correct? Answer only 'True' or 'False'.\n\nClaim: {claim}"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Parse boolean from response
    response_lower = response.lower()
    
    # Look for clear True/False patterns
    if response_lower.startswith("true"):
        return True, response
    elif response_lower.startswith("false"):
        return False, response
    elif "true" in response_lower[:50] and "false" not in response_lower[:50]:
        return True, response
    elif "false" in response_lower[:50] and "true" not in response_lower[:50]:
        return False, response
    elif "true" in response_lower and "false" not in response_lower:
        return True, response
    elif "false" in response_lower and "true" not in response_lower:
        return False, response
    else:
        return None, response

def batch_inference(model, tokenizer, input_file, output_file):
    """Process the test file and generate predictions"""
    
    # Load existing results if resuming
    processed_indices = set()
    results = []
    correct = 0
    total = 0
    
    # Track single-hop and two-hop accuracy
    single_hop_correct = 0
    single_hop_total = 0
    two_hop_correct = 0
    two_hop_total = 0
    
    if os.path.exists(output_file):
        print(f"Found existing results at {output_file}, resuming...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                    processed_indices.add(idx)
                    
                    # Recalculate accuracy
                    predicted = result.get('prediction')
                    ground_truth = result.get('label')
                    
                    # Handle both 'reasoning_type' and 'hop_count' fields
                    reasoning_type = result.get('reasoning_type')
                    if reasoning_type is None:
                        hop_count = result.get('hop_count', 1)
                        reasoning_type = 'single-hop' if hop_count == 1 else 'two-hop'
                    
                    if ground_truth is not None and predicted is not None:
                        total += 1
                        if predicted == ground_truth:
                            correct += 1
                        
                        # Track by reasoning type
                        if reasoning_type == 'single-hop':
                            single_hop_total += 1
                            if predicted == ground_truth:
                                single_hop_correct += 1
                        elif reasoning_type == 'two-hop':
                            two_hop_total += 1
                            if predicted == ground_truth:
                                two_hop_correct += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"Resuming from example {len(results)+1} (already processed {len(results)} examples)")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Open output file in append mode
    output_handle = open(output_file, 'a', encoding='utf-8')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        print(f"Processing {len(all_lines)} examples...")
        
        for idx, line in enumerate(tqdm(all_lines, desc="Benchmarking Llama-3.2-3B")):
            # Skip already processed examples
            if idx in processed_indices:
                continue
            
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            
            # Parse JSON with error handling
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n[ERROR] Failed to parse JSON at line {idx+1}: {e}")
                continue
            
            # Extract fields
            claim = data.get('input_question', data.get('claim', data.get('input', '')))
            ground_truth = data.get('label', None)
            hop_count = data.get('hop_count', 1)
            
            # Generate prediction
            predicted_label, raw_response = verify_claim(model, tokenizer, claim)
            
            # Calculate accuracy
            if ground_truth is not None and predicted_label is not None:
                total += 1
                is_correct = (predicted_label == ground_truth)
                if is_correct:
                    correct += 1
                
                # Track by reasoning type
                reasoning_type = 'single-hop' if hop_count == 1 else 'two-hop'
                
                if reasoning_type == 'single-hop':
                    single_hop_total += 1
                    if is_correct:
                        single_hop_correct += 1
                elif reasoning_type == 'two-hop':
                    two_hop_total += 1
                    if is_correct:
                        two_hop_correct += 1
            
            # Create result entry
            result = {
                **data,
                'prediction': predicted_label,
                'raw_response': raw_response,
                'correct': predicted_label == ground_truth if ground_truth is not None else None,
                'reasoning_type': 'single-hop' if hop_count == 1 else 'two-hop'
            }
            
            # Save result immediately
            results.append(result)
            output_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_handle.flush()
    
    finally:
        output_handle.close()
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"Llama-3.2-3B (Vanilla) Benchmarking Results:")
    print(f"{'='*70}")
    
    if total > 0:
        accuracy = 100 * correct / total
        single_hop_accuracy = 100 * single_hop_correct / single_hop_total if single_hop_total > 0 else 0
        two_hop_accuracy = 100 * two_hop_correct / two_hop_total if two_hop_total > 0 else 0
        
        print(f"\nOverall Performance:")
        print(f"  Total Examples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {total - correct}")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        
        if single_hop_total > 0:
            print(f"\nSingle-Hop Reasoning:")
            print(f"  Total Examples: {single_hop_total}")
            print(f"  Correct: {single_hop_correct}")
            print(f"  Incorrect: {single_hop_total - single_hop_correct}")
            print(f"  Single-Hop Accuracy: {single_hop_accuracy:.2f}%")
        
        if two_hop_total > 0:
            print(f"\nTwo-Hop Reasoning:")
            print(f"  Total Examples: {two_hop_total}")
            print(f"  Correct: {two_hop_correct}")
            print(f"  Incorrect: {two_hop_total - two_hop_correct}")
            print(f"  Two-Hop Accuracy: {two_hop_accuracy:.2f}%")
    
    print(f"{'='*70}\n")
    print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-3B Vanilla Inference Script")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--input", type=str, default="./llm_test.jsonl",
                        help="Input test file (JSONL)")
    parser.add_argument("--output", type=str, default="results/llama32_vanilla_results.jsonl",
                        help="Output results file (JSONL)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Run benchmarking
    batch_inference(model, tokenizer, args.input, args.output)

if __name__ == "__main__":
    main()
