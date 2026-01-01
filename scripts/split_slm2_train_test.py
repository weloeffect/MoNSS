"""
Script to split SLM2 training data into train and test sets.
"""
import json
import random
from pathlib import Path

# Configuration
RANDOM_SEED = 42
TEST_RATIO = 0.2  # 20% for test set

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "train"
INPUT_FILE = DATA_DIR / "slm2_train_data_norm.jsonl"
TRAIN_OUTPUT = DATA_DIR / "slm2_train.jsonl"
TEST_OUTPUT = DATA_DIR / "slm2_test.jsonl"

def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: list[dict], filepath: Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Load data
    print(f"Loading data from {INPUT_FILE}...")
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} records")
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split index
    test_size = int(len(data) * TEST_RATIO)
    train_size = len(data) - test_size
    
    # Split data
    test_data = data[:test_size]
    train_data = data[test_size:]
    
    print(f"Split: {train_size} train, {test_size} test")
    
    # Save splits
    save_jsonl(train_data, TRAIN_OUTPUT)
    print(f"Saved training data to {TRAIN_OUTPUT}")
    
    save_jsonl(test_data, TEST_OUTPUT)
    print(f"Saved test data to {TEST_OUTPUT}")
    
    # Print sample from test set
    print("\n--- Sample from test set ---")
    if test_data:
        sample = test_data[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input'][:150]}...")
        print(f"Output: {sample['output']}")
        print(f"Entities: {sample.get('entities', {})}")

if __name__ == "__main__":
    main()
