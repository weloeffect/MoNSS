#!/usr/bin/env python3
"""
Data Preparation Script for SLM1 (Text2SPARQL) Benchmarking

This script performs:
1. Combines single-hop and multi-hop training data
2. Stratified shuffle to ensure diverse question types
3. 85/15 train/test split
4. Saves to slm1_train_final.jsonl and slm1_test.jsonl
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# File paths
DATA_DIR = Path("./data/train")
SINGLEHOP_FILE = DATA_DIR / "text2sparql_singlehop.jsonl"
MULTIHOP_FILE = DATA_DIR / "text2sparql_multihop.jsonl"
TRAIN_OUTPUT = DATA_DIR / "slm1_train_final.jsonl"
TEST_OUTPUT = DATA_DIR / "slm1_test.jsonl"

# Split ratio
TEST_RATIO = 0.15
TRAIN_RATIO = 0.85


def load_jsonl(filepath):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON at {filepath}:{line_num}")
                print(f"   Error: {e}")
    return data


def save_jsonl(data, filepath):
    """Save list of dictionaries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def classify_query_type(item):
    """
    Classify query as single-hop or multi-hop based on SPARQL structure.
    Returns: 'single-hop' or 'multi-hop'
    """
    sparql = item.get('output', '')
    
    # Count the number of triple patterns (lines with "dbo:" or "dbr:")
    triple_count = sparql.count('dbo:') + sparql.count('dbr:')
    
    # Multi-hop queries typically have intermediate variables (?mid, ?x, etc.)
    has_intermediate_var = '?mid' in sparql or '?x ' in sparql or '?y ' in sparql
    
    # Count SELECT variables vs WHERE clause complexity
    where_clauses = sparql.count(' . ')
    
    if where_clauses >= 2 or has_intermediate_var:
        return 'multi-hop'
    else:
        return 'single-hop'


def stratified_split(data, test_ratio=0.15):
    """
    Perform stratified split to maintain single-hop/multi-hop distribution.
    
    Args:
        data: List of data items
        test_ratio: Proportion of data for test set
    
    Returns:
        train_set, test_set
    """
    # Group data by query type
    stratified_groups = defaultdict(list)
    
    for item in data:
        query_type = classify_query_type(item)
        stratified_groups[query_type].append(item)
    
    print(f"\n📊 Data Distribution:")
    for query_type, items in stratified_groups.items():
        print(f"   {query_type}: {len(items)} samples")
    
    # Shuffle each group independently
    for query_type in stratified_groups:
        random.shuffle(stratified_groups[query_type])
    
    # Split each group
    train_set = []
    test_set = []
    
    for query_type, items in stratified_groups.items():
        split_idx = int(len(items) * (1 - test_ratio))
        train_set.extend(items[:split_idx])
        test_set.extend(items[split_idx:])
        
        print(f"   {query_type} → Train: {split_idx}, Test: {len(items) - split_idx}")
    
    # Final shuffle to mix query types
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set


def main():
    print("=" * 80)
    print("SLM1 Data Preparation for Benchmarking")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    
    if not SINGLEHOP_FILE.exists():
        print(f"❌ Error: {SINGLEHOP_FILE} not found!")
        return
    
    if not MULTIHOP_FILE.exists():
        print(f"❌ Error: {MULTIHOP_FILE} not found!")
        return
    
    singlehop_data = load_jsonl(SINGLEHOP_FILE)
    multihop_data = load_jsonl(MULTIHOP_FILE)
    
    print(f"   ✅ Loaded {len(singlehop_data)} single-hop examples")
    print(f"   ✅ Loaded {len(multihop_data)} multi-hop examples")
    
    # Step 2: Combine and deduplicate data
    print("\n🔗 Step 2: Combining and deduplicating datasets...")
    combined_data = singlehop_data + multihop_data
    
    # Remove duplicates based on input question
    seen_questions = set()
    unique_data = []
    duplicates_removed = 0
    
    for item in combined_data:
        question = item.get('input', '')
        if question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)
        else:
            duplicates_removed += 1
    
    combined_data = unique_data
    total_samples = len(combined_data)
    
    if duplicates_removed > 0:
        print(f"   ⚠️  Removed {duplicates_removed} duplicate questions")
    print(f"   ✅ Total unique samples: {total_samples}")
    print(f"   ✅ Total samples: {total_samples}")
    
    # Step 3: Stratified split
    print("\n🎲 Step 3: Performing stratified shuffle and split...")
    print(f"   Split ratio: {TRAIN_RATIO:.0%} train / {TEST_RATIO:.0%} test")
    
    train_set, test_set = stratified_split(combined_data, test_ratio=TEST_RATIO)
    
    print(f"\n✅ Split complete:")
    print(f"   Training set: {len(train_set)} samples ({len(train_set)/total_samples:.1%})")
    print(f"   Test set: {len(test_set)} samples ({len(test_set)/total_samples:.1%})")
    
    # Step 4: Save data
    print("\n💾 Step 4: Saving datasets...")
    
    save_jsonl(train_set, TRAIN_OUTPUT)
    print(f"   ✅ Saved training data: {TRAIN_OUTPUT}")
    
    save_jsonl(test_set, TEST_OUTPUT)
    print(f"   ✅ Saved test data: {TEST_OUTPUT}")
    
    # Step 5: Verification
    print("\n🔍 Step 5: Verification...")
    
    # Check for data leakage (shouldn't find duplicates between train/test)
    train_queries = {item['input'] for item in train_set}
    test_queries = {item['input'] for item in test_set}
    overlap = train_queries & test_queries
    
    if overlap:
        print(f"   ⚠️  Warning: Found {len(overlap)} duplicate queries between train/test")
    else:
        print(f"   ✅ No data leakage detected")
    
    print("\n" + "=" * 80)
    print("✅ Data preparation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review test set: {TEST_OUTPUT}")
    print(f"2. Use {TRAIN_OUTPUT} for SLM1 training")
    print(f"3. Run 'prepare_slm1_adversarial.py' to generate zero-result queries")
    print("=" * 80)


if __name__ == "__main__":
    main()
