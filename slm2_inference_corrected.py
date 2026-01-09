#!/usr/bin/env python3
"""
SLM2 Inference and Benchmarking Script - CORRECTED VERSION
Copy this entire content to your VM to replace slm2_inference.py
"""
print("Script starting...", flush=True)

import sys
import json
import torch
import re
from pathlib import Path

print("Basic imports done", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

print("All imports done", flush=True)

# Configuration
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
ADAPTER_PATH = Path("outputs/Llama-3-8B-sparql2Text")

# File paths - files are in root directory on VM
TEST_FILE = Path("slm2_test.jsonl")
PREDICTIONS_FILE = Path("slm2_predictions.jsonl")
RESULTS_FILE = Path("slm2_benchmark_results.jsonl")

print("Configuration loaded", flush=True)


def load_model():
    print("=" * 60)
    print("Loading SLM2 Model")
    print("=" * 60)
    print("Base model: " + BASE_MODEL_ID)
    print("Adapter: " + str(ADAPTER_PATH))
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        print("GPU cache cleared")
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model with 4-bit quantization...")
    # Use much more aggressive memory limits
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "3.5GiB", "cpu": "40GiB"},  # Very conservative GPU limit
        low_cpu_mem_usage=True,
        offload_folder="offload",
        torch_dtype=torch.float16,
    )
    
    # Aggressive memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        print(f"Memory after base model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    print("Loading LoRA adapter with CPU offloading...")
    try:
        # Try loading adapter with very conservative settings
        model = PeftModel.from_pretrained(
            base_model, 
            str(ADAPTER_PATH),
            device_map="auto",
            max_memory={0: "3.5GiB", "cpu": "40GiB"},
            offload_folder="offload",
            low_cpu_mem_usage=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ GPU memory exhausted, falling back to CPU-only adapter loading...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Force adapter to CPU first
            model = PeftModel.from_pretrained(
                base_model, 
                str(ADAPTER_PATH),
                device_map={"": "cpu"},  # Force CPU
            )
            print("✅ Adapter loaded on CPU, moving selected parts to GPU...")
        else:
            raise e
    
    model.eval()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Final memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    print("Model loaded successfully!")
    return tokenizer, model


def load_test_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print("Warning: Skipping line " + str(line_num) + ": " + str(e))
    return data


def generate_response(sparql_input, instruction, tokenizer, model):
    prompt = "### Instruction:\n" + instruction + "\n\n### Input:\n" + sparql_input + "\n\n### Output:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "### Output:" in output_text:
        response = output_text.split("### Output:")[-1].strip()
    else:
        response = output_text.strip()
    
    return response.split("\n")[0].strip()


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_entities_from_sparql_result(sparql_input):
    """
    Extract entities from the SPARQL Result section.
    
    Returns: set of entity values
    """
    entities = set()
    
    # Find SPARQL Result section
    result_match = re.search(r'SPARQL Result:\s*\n(.*?)(?:\n\n|$)', sparql_input, re.DOTALL)
    if not result_match:
        return entities
    
    result_section = result_match.group(1)
    
    # Extract values after colons (e.g., "- place: Melbourne" -> "Melbourne")
    pattern = r'-\s+\w+:\s+(.+?)(?=\n|$)'
    matches = re.findall(pattern, result_section)
    
    for match in matches:
        entity = match.strip()
        if entity and entity.lower() not in ['empty', 'none', 'null']:
            entities.add(entity)
    
    return entities


def check_hallucination(predicted, kg_entities):
    """
    Check if predicted output contains entities not in SPARQL results.
    
    Returns: (has_hallucination, matched_entities, hallucinations)
    """
    pred_lower = predicted.lower()
    matched = set()
    
    # Check which KG entities appear in output
    for entity in kg_entities:
        if entity.lower() in pred_lower:
            matched.add(entity)
    
    # Extract capitalized phrases (potential entities)
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    output_entities = set(re.findall(capitalized_pattern, predicted))
    
    # Find hallucinations (entities in output not from KG)
    hallucinations = set()
    for output_entity in output_entities:
        is_from_kg = False
        for kg_entity in kg_entities:
            if (output_entity.lower() in kg_entity.lower() or 
                kg_entity.lower() in output_entity.lower()):
                is_from_kg = True
                break
        
        if not is_from_kg:
            hallucinations.add(output_entity)
    
    has_hallucination = len(hallucinations) > 0
    return has_hallucination, matched, hallucinations


def compute_entity_coverage(matched_entities, kg_entities):
    """
    Compute what % of KG entities appear in the output.
    """
    if not kg_entities:
        return 1.0
    return len(matched_entities) / len(kg_entities)


def compute_metrics(pred, gt, entities, sparql_input):
    exact_match = normalize_text(pred) == normalize_text(gt)
    
    pred_lower = pred.lower()
    if "i don't know" in gt.lower():
        contains_answer = "don't know" in pred_lower
    else:
        contains_answer = True
        for vals in entities.values():
            if isinstance(vals, list):
                for v in vals:
                    if v.lower() not in pred_lower:
                        contains_answer = False
                        break
    
    pred_tokens = set(normalize_text(pred).split())
    gt_tokens = set(normalize_text(gt).split())
    if gt_tokens:
        overlap = len(pred_tokens & gt_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(gt_tokens)
        if (precision + recall) > 0:
            bleu = 2 * precision * recall / (precision + recall)
        else:
            bleu = 0
    else:
        if not pred_tokens:
            bleu = 1.0
        else:
            bleu = 0.0
    
    # NEW: Hallucination detection and entity coverage
    kg_entities = extract_entities_from_sparql_result(sparql_input)
    has_hallucination, matched_entities, hallucinations = check_hallucination(pred, kg_entities)
    entity_coverage = compute_entity_coverage(matched_entities, kg_entities)
    
    return {
        'exact_match': exact_match,
        'contains_answer': contains_answer,
        'bleu_score': round(bleu, 4),
        'has_hallucination': has_hallucination,
        'hallucinations': list(hallucinations),
        'entity_coverage': round(entity_coverage, 4),
        'kg_entities': list(kg_entities),
        'matched_entities': list(matched_entities)
    }


def run_inference(test_data, tokenizer, model):
    predictions = []
    print("\nRunning inference on " + str(len(test_data)) + " examples...")
    
    for i, example in enumerate(tqdm(test_data, desc="Generating")):
        instruction = example.get("instruction", "")
        sparql_input = example.get("input", "")
        ground_truth = example.get("output", "")
        entities = example.get("entities", {})
        
        predicted = generate_response(sparql_input, instruction, tokenizer, model)
        metrics = compute_metrics(predicted, ground_truth, entities, sparql_input)
        
        prediction = {
            "index": i + 1,
            "input": sparql_input,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "entities": entities,
            "metrics": metrics
        }
        predictions.append(prediction)
        
        if i < 3:
            print("\n[Example " + str(i + 1) + "]")
            print("Ground Truth: " + ground_truth)
            print("Predicted: " + predicted)
            print("Exact Match: " + str(metrics['exact_match']))
            print("Hallucination: " + str(metrics['has_hallucination']))
            print("Entity Coverage: " + str(metrics['entity_coverage']))
    
    return predictions


def compute_aggregate_metrics(predictions):
    total = len(predictions)
    exact = sum(1 for p in predictions if p["metrics"]["exact_match"])
    contains = sum(1 for p in predictions if p["metrics"]["contains_answer"])
    avg_bleu = sum(p["metrics"]["bleu_score"] for p in predictions) / total if total else 0
    
    # NEW: Hallucination metrics
    hallucination_count = sum(1 for p in predictions if p["metrics"]["has_hallucination"])
    coverage_scores = [p["metrics"]["entity_coverage"] for p in predictions]
    avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
    
    idk_total = 0
    idk_correct = 0
    for p in predictions:
        gt = p["ground_truth"].lower()
        pred = p["predicted"].lower()
        if "i don't know" in gt:
            idk_total += 1
            if "don't know" in pred:
                idk_correct += 1
    
    if total > 0:
        exact_rate = round(exact / total, 4)
        contains_rate = round(contains / total, 4)
        hallucination_rate = round(hallucination_count / total, 4)
    else:
        exact_rate = 0
        contains_rate = 0
        hallucination_rate = 0
    
    if idk_total > 0:
        idk_rate = round(idk_correct / idk_total, 4)
    else:
        idk_rate = 1.0
    
    return {
        "total_examples": total,
        "exact_match": {"count": exact, "rate": exact_rate},
        "contains_answer": {"count": contains, "rate": contains_rate},
        "avg_bleu_score": round(avg_bleu, 4),
        "hallucination": {"count": hallucination_count, "rate": hallucination_rate},
        "entity_coverage": {"avg": round(avg_coverage, 4), "scores": coverage_scores},
        "idk_handling": {"total": idk_total, "correct": idk_correct, "rate": idk_rate}
    }


def main():
    print("Entering main...", flush=True)
    print("=" * 60)
    print("SLM2 Inference and Benchmarking")
    print("=" * 60)
    
    print("Test file: " + str(TEST_FILE.absolute()))
    print("Adapter path: " + str(ADAPTER_PATH.absolute()))
    
    if not TEST_FILE.exists():
        print("ERROR: Test file not found: " + str(TEST_FILE.absolute()))
        sys.exit(1)
    
    if not ADAPTER_PATH.exists():
        print("ERROR: Adapter not found: " + str(ADAPTER_PATH.absolute()))
        sys.exit(1)
    
    tokenizer, model = load_model()
    test_data = load_test_data(TEST_FILE)
    print("\nLoaded " + str(len(test_data)) + " test examples")
    
    predictions = run_inference(test_data, tokenizer, model)
    
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print("\nSaved predictions to: " + str(PREDICTIONS_FILE))
    
    metrics = compute_aggregate_metrics(predictions)
    
    # Save results as JSONL (one JSON object per line)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"type": "metadata", "model": str(ADAPTER_PATH)}, ensure_ascii=False) + '\n')
        f.write(json.dumps({"type": "metrics", **metrics}, ensure_ascii=False) + '\n')
    print("Saved results to: " + str(RESULTS_FILE))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("Total: " + str(metrics['total_examples']))
    print("Exact Match: " + str(round(metrics['exact_match']['rate'] * 100, 2)) + "%")
    print("Contains Answer: " + str(round(metrics['contains_answer']['rate'] * 100, 2)) + "%")
    print("Avg BLEU: " + str(metrics['avg_bleu_score']))
    print("\n--- Hallucination & Entity Coverage ---")
    print("Hallucination Rate: " + str(round(metrics['hallucination']['rate'] * 100, 2)) + "%")
    print("Avg Entity Coverage: " + str(round(metrics['entity_coverage']['avg'] * 100, 2)) + "%")
    print("\n--- Critical Failure Constraint ---")
    print("IDK Handling: " + str(round(metrics['idk_handling']['rate'] * 100, 2)) + "%")
    print("=" * 60)
    print("Benchmarking complete!")


if __name__ == "__main__":
    main()