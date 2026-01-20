"""
Inference script for Microsoft Phi-3-medium-4k-instruct
Can be used for both Text->SPARQL and SPARQL->Text tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json

def load_model(model_name="microsoft/Phi-3-medium-4k-instruct", device="cuda"):
    """Load Phi-3 model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print(f"Model loaded on {device}")
    return model, tokenizer

def format_phi3_prompt(instruction, input_text, system_msg=None):
    """Format prompt for Phi-3 using its chat template"""
    if system_msg is None:
        system_msg = "You are a helpful AI assistant."
    
    # Phi-3 chat template
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"}
    ]
    
    return messages

def generate_text2sparql(model, tokenizer, question, schema=None, max_length=512):
    """Generate SPARQL query from natural language question"""
    system_msg = "You are a helpful assistant that converts natural language questions into SPARQL queries using the DBpedia ontology."
    
    instruction = "Convert the following natural language question into a SPARQL query."
    
    if schema:
        input_text = f"Schema:\n{schema}\n\nQuestion:\n{question}"
    else:
        input_text = f"Question:\n{question}"
    
    messages = format_phi3_prompt(instruction, input_text, system_msg)
    
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
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response

def generate_sparql2text(model, tokenizer, triples, max_length=256):
    """Generate natural language from SPARQL triples"""
    system_msg = "You are a helpful assistant that converts knowledge graph triples into natural language sentences."
    
    instruction = "Generate a natural language sentence that expresses the following knowledge graph facts."
    
    # Format triples
    if isinstance(triples, str):
        input_text = f"Knowledge Graph Facts:\n{triples}"
    else:
        input_text = f"Knowledge Graph Facts:\n{json.dumps(triples)}"
    
    messages = format_phi3_prompt(instruction, input_text, system_msg)
    
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
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response

def verify_claim(model, tokenizer, claim, max_length=128):
    """Verify if a claim is true or false"""
    system_msg = "You are a helpful assistant that verifies factual claims. Answer only with 'True' or 'False'."
    
    instruction = "Is the following claim factually correct? Answer only 'True' or 'False'."
    input_text = f"Claim: {claim}"
    
    messages = format_phi3_prompt(instruction, input_text, system_msg)
    
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
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Parse boolean from response - search entire response, prioritize early matches
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

def batch_inference(model, tokenizer, input_file, output_file, task="text2sparql", save_every=10):
    """Process a batch of inputs from a JSONL file with checkpoint support"""
    import os
    
    # Load existing results if resuming
    processed_indices = set()
    results = []
    correct = 0
    total = 0
    
    # Track single-hop and two-hop accuracy separately
    single_hop_correct = 0
    single_hop_total = 0
    two_hop_correct = 0
    two_hop_total = 0
    
    if os.path.exists(output_file):
        print(f"Found existing results at {output_file}, resuming...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                result = json.loads(line.strip())
                results.append(result)
                processed_indices.add(idx)
                
                # Recalculate accuracy for verify task
                if task == "verify":
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
        
        print(f"Resuming from example {len(results)+1} (already processed {len(results)} examples)")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open output file in append mode
    output_handle = open(output_file, 'a', encoding='utf-8')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # Skip already processed examples
                if idx in processed_indices:
                    continue
                
                data = json.loads(line.strip())
                
                if task == "text2sparql":
                    question = data.get('input', data.get('question', ''))
                    schema = data.get('schema', None)
                    prediction = generate_text2sparql(model, tokenizer, question, schema)
                    result = {
                        **data,
                        'prediction': prediction
                    }
                    results.append(result)
                    output_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_handle.flush()
                    print(f"Processed {len(results)} examples", end='\r')
                    
                elif task == "sparql2text":
                    triples = data.get('input', data.get('triples', ''))
                    prediction = generate_sparql2text(model, tokenizer, triples)
                    result = {
                        **data,
                        'prediction': prediction
                    }
                    results.append(result)
                    output_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_handle.flush()
                    print(f"Processed {len(results)} examples", end='\r')
                    
                elif task == "verify":
                    claim = data.get('input_question', data.get('claim', data.get('input', '')))
                    predicted_label, raw_response = verify_claim(model, tokenizer, claim)
                    
                    # Get ground truth label and reasoning type
                    ground_truth = data.get('label', None)
                    
                    # Handle both 'reasoning_type' and 'hop_count' fields
                    reasoning_type = data.get('reasoning_type')
                    if reasoning_type is None:
                        hop_count = data.get('hop_count', 1)
                        reasoning_type = 'single-hop' if hop_count == 1 else 'two-hop'
                    
                    # Calculate accuracy
                    if ground_truth is not None and predicted_label is not None:
                        total += 1
                        is_correct = (predicted_label == ground_truth)
                        if is_correct:
                            correct += 1
                        
                        # Track by reasoning type
                        if reasoning_type == 'single-hop':
                            single_hop_total += 1
                            if is_correct:
                                single_hop_correct += 1
                        elif reasoning_type == 'two-hop':
                            two_hop_total += 1
                            if is_correct:
                                two_hop_correct += 1
                    elif predicted_label is None:
                        # Debug: print unparseable responses (first 10 only)
                        if len(results) < 10:
                            print(f"\n[DEBUG] Could not parse response for example {len(results)+1}: {raw_response[:100]}...")
                    
                    result = {
                    **data,
                    'prediction': predicted_label,
                    'raw_response': raw_response,
                    'correct': predicted_label == ground_truth if ground_truth is not None else None,
                    'reasoning_type': reasoning_type
                }
                
                # Append result immediately to file
                results.append(result)
                output_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_handle.flush()  # Force write to disk
                
                # Print progress
                if task == "verify":
                    print(f"Processed {len(results)} examples | Overall: {correct}/{total} ({100*correct/total if total > 0 else 0:.2f}%) | "
                          f"Single-hop: {single_hop_correct}/{single_hop_total} ({100*single_hop_correct/single_hop_total if single_hop_total > 0 else 0:.2f}%) | "
                          f"Two-hop: {two_hop_correct}/{two_hop_total} ({100*two_hop_correct/two_hop_total if two_hop_total > 0 else 0:.2f}%)", end='\r')
                else:
                    print(f"Processed {len(results)} examples", end='\r')
    
    finally:
        output_handle.close()
    
    print(f"\nCompleted {len(results)} examples")
    
    # Print accuracy for verification task
    if task == "verify" and total > 0:
        accuracy = 100 * correct / total
        single_hop_accuracy = 100 * single_hop_correct / single_hop_total if single_hop_total > 0 else 0
        two_hop_accuracy = 100 * two_hop_correct / two_hop_total if two_hop_total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Phi-3 Benchmarking Results:")
        print(f"{'='*70}")
        print(f"\nOverall Performance:")
        print(f"  Total Examples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {total - correct}")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        
        print(f"\nSingle-Hop Reasoning:")
        print(f"  Total Examples: {single_hop_total}")
        print(f"  Correct: {single_hop_correct}")
        print(f"  Incorrect: {single_hop_total - single_hop_correct}")
        print(f"  Single-Hop Accuracy: {single_hop_accuracy:.2f}%")
        
        print(f"\nTwo-Hop Reasoning:")
        print(f"  Total Examples: {two_hop_total}")
        print(f"  Correct: {two_hop_correct}")
        print(f"  Incorrect: {two_hop_total - two_hop_correct}")
        print(f"  Two-Hop Accuracy: {two_hop_accuracy:.2f}%")
        print(f"{'='*70}\n")
    
    print(f"Results saved to {output_file}")

def interactive_mode(model, tokenizer, task="text2sparql"):
    """Interactive mode for testing"""
    print(f"\n{'='*60}")
    print(f"Phi-3 Interactive Mode - {task.upper()}")
    print(f"{'='*60}")
    print("Type 'exit' to quit\n")
    
    while True:
        if task == "text2sparql":
            question = input("\nEnter a natural language question: ").strip()
            if question.lower() == 'exit':
                break
            
            result = generate_text2sparql(model, tokenizer, question)
            print(f"\nGenerated SPARQL:\n{result}")
        
        elif task == "sparql2text":
            triples = input("\nEnter knowledge graph triples (JSON or text): ").strip()
            if triples.lower() == 'exit':
                break
            
            result = generate_sparql2text(model, tokenizer, triples)
            print(f"\nGenerated Text:\n{result}")
        
        elif task == "verify":
            claim = input("\nEnter a claim to verify: ").strip()
            if claim.lower() == 'exit':
                break
            
            prediction, raw_response = verify_claim(model, tokenizer, claim)
            print(f"\nPrediction: {prediction}")
            print(f"Raw Response: {raw_response}")

def main():
    parser = argparse.ArgumentParser(description="Phi-3 Inference Script")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-medium-4k-instruct",
                        help="Model name or path")
    parser.add_argument("--task", type=str, choices=["text2sparql", "sparql2text", "verify"],
                        default="verify", help="Task type")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"],
                        default="batch", help="Inference mode")
    parser.add_argument("--input", type=str, default="llm_test.jsonl",
                        help="Input JSONL file for batch mode")
    parser.add_argument("--output", type=str, default="results/phi3_verification_results.jsonl",
                        help="Output JSONL file for batch mode")
    parser.add_argument("--question", type=str, help="Single question for direct inference")
    parser.add_argument("--triples", type=str, help="Single triples for direct inference")
    parser.add_argument("--claim", type=str, help="Single claim to verify")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Run inference
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args.task)
    elif args.mode == "batch":
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            return
        batch_inference(model, tokenizer, args.input, args.output, args.task)
    else:
        # Direct inference mode with single inputs
        if args.question:
            result = generate_text2sparql(model, tokenizer, args.question)
            print(f"\nQuestion: {args.question}")
            print(f"\nGenerated SPARQL:\n{result}")
        elif args.triples:
            result = generate_sparql2text(model, tokenizer, args.triples)
            print(f"\nTriples: {args.triples}")
            print(f"\nGenerated Text:\n{result}")
        elif args.claim:
            prediction, raw_response = verify_claim(model, tokenizer, args.claim)
            print(f"\nClaim: {args.claim}")
            print(f"\nPrediction: {prediction}")
            print(f"Raw Response: {raw_response}")
        else:
            print("Error: Please specify --question, --triples, or --claim for direct inference")

if __name__ == "__main__":
    main()