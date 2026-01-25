"""
Analyze SLM2 benchmark results to understand what went wrong
"""

import json
from collections import Counter

def analyze_results(results_file):
    predictions = []
    expected = []
    correct_examples = []
    incorrect_examples = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            
            data = json.loads(line.strip())
            pred = data.get('prediction', '')
            exp = data.get('output', '')
            is_correct = data.get('correct', False)
            
            predictions.append(pred)
            expected.append(exp)
            
            if is_correct:
                correct_examples.append((idx, exp, pred))
            else:
                incorrect_examples.append((idx, exp, pred))
    
    print("="*70)
    print("SLM2 Results Analysis")
    print("="*70)
    
    # Count unique predictions
    pred_counter = Counter(predictions)
    exp_counter = Counter(expected)
    
    print("\nMost common predictions:")
    for pred, count in pred_counter.most_common(10):
        print(f"  {count:4d}x: {pred[:100]}")
    
    print("\nMost common expected outputs:")
    for exp, count in exp_counter.most_common(10):
        print(f"  {count:4d}x: {exp[:100]}")
    
    print(f"\nTotal examples: {len(predictions)}")
    print(f"Correct: {len(correct_examples)}")
    print(f"Incorrect: {len(incorrect_examples)}")
    
    print("\n" + "="*70)
    print("Sample CORRECT predictions:")
    print("="*70)
    for idx, exp, pred in correct_examples[:5]:
        print(f"\nExample {idx+1}:")
        print(f"  Expected: {exp}")
        print(f"  Predicted: {pred}")
    
    print("\n" + "="*70)
    print("Sample INCORRECT predictions:")
    print("="*70)
    for idx, exp, pred in incorrect_examples[:10]:
        print(f"\nExample {idx+1}:")
        print(f"  Expected: {exp}")
        print(f"  Predicted: {pred}")

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else "results/slm2_benchmark_results.jsonl"
    analyze_results(results_file)
