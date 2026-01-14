"""
Quick test script to verify GPT-OSS inference setup.
Tests on a small subset of examples before running full benchmark.
"""

import json
from pathlib import Path
from gpt_oss_inference import GPTOSSInference


def test_inference():
    """Test inference on a few examples."""
    
    print("="*70)
    print(" GPT-OSS Inference Test")
    print("="*70)
    
    # Check API key
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n❌ OPENROUTER_API_KEY not set!")
        print("\nSet it with:")
        print("  PowerShell: $env:OPENROUTER_API_KEY = 'your-key-here'")
        print("  CMD:        set OPENROUTER_API_KEY=your-key-here")
        print("  Bash:       export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print(f"\n✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Load a few test examples
    test_file = Path(__file__).parent / '../data/test/slm2_test.jsonl'
    
    if not test_file.exists():
        print(f"\n❌ Test file not found: {test_file}")
        return
    
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Only load 5 examples
                break
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"\n✅ Loaded {len(examples)} test examples")
    
    # Initialize inference
    try:
        inference = GPTOSSInference(api_key=api_key)
        print("\n✅ GPT-OSS client initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize client: {e}")
        return
    
    # Test inference on examples
    print("\n" + "="*70)
    print(" Running Test Predictions")
    print("="*70)
    
    for i, example in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {example['input']}")
        print(f"Gold:  {example['output']}")
        print(f"Label: {example['label']}, Hops: {example['hop_count']}")
        
        try:
            prediction = inference.get_prediction(
                example['instruction'],
                example['input'],
                use_reasoning=False
            )
            print(f"Pred:  {prediction}")
            
            # Check correctness
            if example['label'][0] == False:
                is_correct = prediction.strip() == "I don't know."
            else:
                is_correct = prediction.strip() == example['output'].strip()
            
            print(f"✅ CORRECT" if is_correct else "❌ INCORRECT")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("\nIf the test worked, you can run the full benchmark with:")
    print("  python gpt_oss_inference.py")
    print("="*70)


if __name__ == '__main__':
    test_inference()
