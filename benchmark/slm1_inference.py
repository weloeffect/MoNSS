#!/usr/bin/env python3
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_BASE = "meta-llama/Meta-Llama-3-8B"
ADAPTER_PATH = "./outputs/Llama-3-8B-text2sparql-qlora"
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "predictions"

TEST_FILES = {
    "standard": BENCHMARK_DIR / "slm1_test.jsonl",
    "adversarial": BENCHMARK_DIR / "slm1_adversarial_test.jsonl"
}


def load_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully")
    return tokenizer, model


def generate_sparql(question, context, tokenizer, model):
    prompt = "### Instruction:\nTranslate the question into a SPARQL query.\n\n"
    prompt = prompt + "### Schema:\n" + context + "\n\n"
    prompt = prompt + "### Question:\n" + question + "\n\n"
    prompt = prompt + "### SPARQL:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "### SPARQL:" in output_text:
        sparql = output_text.split("### SPARQL:")[-1].strip()
    else:
        sparql = output_text.strip()
    
    return sparql


def load_test_data(filepath):
    data = []
    skipped = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print("Warning: Skipping malformed line " + str(line_num) + ": " + str(e))
                    skipped = skipped + 1
    if skipped > 0:
        print("Total skipped lines: " + str(skipped))
    return data


def generate_predictions_for_dataset(dataset_name, test_file, tokenizer, model):
    print("=" * 80)
    print("Generating predictions for: " + dataset_name)
    print("=" * 80)
    
    test_data = load_test_data(test_file)
    print("Loaded " + str(len(test_data)) + " test examples")
    
    if len(test_data) == 0:
        print("No valid test data found, skipping...")
        return []
    
    predictions = []
    
    for i, example in enumerate(tqdm(test_data, desc="Generating"), 1):
        question = example.get("input", "")
        if not question:
            print("Warning: Empty question at index " + str(i) + ", skipping...")
            continue
            
        context = example.get("context", "Schema: Film --director--> Person")
        ground_truth = example.get("output", "")
        
        predicted_sparql = generate_sparql(question, context, tokenizer, model)
        
        prediction = {
            "index": i,
            "question": question,
            "context": context,
            "ground_truth_sparql": ground_truth,
            "predicted_sparql": predicted_sparql,
            "adversarial": example.get("adversarial", False),
            "expected_result": example.get("expected_result", None)
        }
        
        predictions.append(prediction)
        
        if i <= 3:
            print("[Example " + str(i) + "]")
            print("Question: " + question)
            pred_display = predicted_sparql[:100] if len(predicted_sparql) > 100 else predicted_sparql
            print("Predicted: " + pred_display)
    
    return predictions


def save_predictions(predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"total_predictions": len(predictions), "predictions": predictions}, f, indent=2)
    print("Saved " + str(len(predictions)) + " predictions to: " + str(output_file))


def main():
    print("=" * 80)
    print("SLM1 Prediction Generation for Benchmarking")
    print("=" * 80)
    print("Working directory: " + str(BENCHMARK_DIR))
    print("Output directory: " + str(OUTPUT_DIR))
    print("Adapter path: " + ADAPTER_PATH)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    adapter_path = Path(ADAPTER_PATH)
    print("Checking for model at: " + str(adapter_path.absolute()))
    print("Model exists: " + str(adapter_path.exists()))
    
    if not adapter_path.exists():
        print("Error: Model not found at " + ADAPTER_PATH)
        print("Train the model first using: python train_main.py")
        return
    
    print("Checking test files...")
    missing_files = []
    for name in TEST_FILES:
        path = TEST_FILES[name]
        exists = path.exists()
        status = "Found" if exists else "MISSING"
        print("  " + name + ": " + path.name + " - " + status)
        if not exists:
            missing_files.append(name)
    
    if len(missing_files) > 0:
        print("Error: Missing test files:")
        for name in missing_files:
            print("   - " + str(TEST_FILES[name].name))
        return
    
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    tokenizer, model = load_model()
    
    all_predictions = {}
    
    for dataset_name in TEST_FILES:
        test_file = TEST_FILES[dataset_name]
        if not test_file.exists():
            print("Warning: " + str(test_file) + " not found, skipping...")
            continue
        
        predictions = generate_predictions_for_dataset(dataset_name, test_file, tokenizer, model)
        
        if len(predictions) > 0:
            output_file = OUTPUT_DIR / ("slm1_predictions_" + dataset_name + ".json")
            save_predictions(predictions, output_file)
            all_predictions[dataset_name] = predictions
        else:
            print("No predictions generated for " + dataset_name)
    
    print("=" * 80)
    print("Prediction Generation Summary")
    print("=" * 80)
    
    for dataset_name in all_predictions:
        predictions = all_predictions[dataset_name]
        print(dataset_name.capitalize() + ": " + str(len(predictions)) + " predictions")
    
    print("=" * 80)
    print("Prediction generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()