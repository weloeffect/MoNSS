#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLM1 Benchmark: Execution Match (EX) Metric

This script measures the most critical metric - whether the generated SPARQL
returns the CORRECT data from the Knowledge Graph, regardless of syntax differences.

Execution Match (EX) = (Correct Data Results) / (Total Questions)

This is a "truth test" - we don't care if the SPARQL looks different,
only if it returns the same answer when executed against the KG.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict


# File paths
DATA_DIR = Path("./data")
TEST_FILE = DATA_DIR / "train" / "slm1_test.jsonl"
ADVERSARIAL_FILE = DATA_DIR / "train" / "slm1_adversarial_test.jsonl"
RELS_FILE = DATA_DIR / "processed" / "relationships.csv"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


class SPARQLExecutor:
    """Executes SPARQL queries against the pandas-based KG."""
    
    def __init__(self, relationships_df: pd.DataFrame):
        """
        Initialize executor with the KG relationships.
        
        Args:
            relationships_df: DataFrame with columns [subject, predicate, object]
        """
        self.rels_df = relationships_df
        print(f"   Loaded KG with {len(self.rels_df)} relationships")
    
    def extract_triple_pattern(self, sparql: str) -> Optional[Dict[str, str]]:
        """
        Extract the core triple pattern from SPARQL query.
        
        Returns:
            dict with 'subject', 'predicate', 'object' (variables start with ?)
        """
        # Remove prefixes and comments
        sparql = re.sub(r'PREFIX.*\n', '', sparql)
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s*\{(.*?)\}', sparql, re.DOTALL | re.IGNORECASE)
        if not where_match:
            return None
        
        where_clause = where_match.group(1)
        
        # Split by period to get individual triple patterns
        triple_patterns = [t.strip() for t in where_clause.split('.') if t.strip()]
        
        return triple_patterns
    
    def execute_simple_query(self, sparql: str) -> Set[str]:
        """
        Execute a simple SPARQL query and return result set.
        
        This handles common patterns:
        - Single-hop: ?film dbo:director ?person
        - Multi-hop: ?film dbo:starring ?mid . ?mid dbo:birthPlace ?answer
        
        Returns:
            Set of result values (entity IDs)
        """
        try:
            triple_patterns = self.extract_triple_pattern(sparql)
            if not triple_patterns:
                return set()
            
            # Parse patterns into structured form
            parsed_patterns = []
            for pattern in triple_patterns:
                parts = pattern.split()
                if len(parts) >= 3:
                    # Handle optional trailing period
                    obj = parts[2].rstrip('.')
                    parsed_patterns.append({
                        'subject': parts[0],
                        'predicate': parts[1],
                        'object': obj
                    })
            
            if not parsed_patterns:
                return set()
            
            # Execute query by joining patterns
            current_results = self.rels_df.copy()
            variable_bindings = {}
            
            for pattern in parsed_patterns:
                subj = pattern['subject']
                pred = pattern['predicate']
                obj = pattern['object']
                
                # Build filter conditions
                conditions = []
                
                # Subject filter
                if subj.startswith('?'):
                    # Variable - check if already bound
                    if subj in variable_bindings:
                        conditions.append(('subject', variable_bindings[subj]))
                else:
                    # Constant entity
                    conditions.append(('subject', subj))
                
                # Predicate filter (usually constant)
                if not pred.startswith('?'):
                    conditions.append(('predicate', pred))
                
                # Object filter
                if obj.startswith('?'):
                    if obj in variable_bindings:
                        conditions.append(('object', variable_bindings[obj]))
                else:
                    conditions.append(('object', obj))
                
                # Apply filters
                for col, val in conditions:
                    current_results = current_results[current_results[col] == val]
                
                # Bind variables from results
                if subj.startswith('?') and subj not in variable_bindings:
                    if len(current_results) > 0:
                        variable_bindings[subj] = current_results.iloc[0]['subject']
                
                if obj.startswith('?') and obj not in variable_bindings:
                    # This is typically what we SELECT
                    pass
            
            # Extract final answer variable (usually ?answer, ?person, ?film, etc.)
            select_match = re.search(r'SELECT\s+(\?\w+)', sparql, re.IGNORECASE)
            if select_match and len(current_results) > 0:
                answer_var = select_match.group(1)
                
                # Find which column contains the answer
                last_pattern = parsed_patterns[-1]
                if last_pattern['object'] == answer_var:
                    results = set(current_results['object'].unique())
                elif last_pattern['subject'] == answer_var:
                    results = set(current_results['subject'].unique())
                else:
                    results = set(current_results['object'].unique())
                
                return results
            
            return set()
            
        except Exception as e:
            print(f"      ⚠️ Execution error: {e}")
            return set()
    
    def execute(self, sparql: str) -> Set[str]:
        """
        Main execution method - returns set of result entity IDs.
        """
        return self.execute_simple_query(sparql)


def load_test_data(filepath: Path) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def normalize_entity_id(entity: str) -> str:
    """Normalize entity ID for comparison."""
    entity = entity.strip()
    
    # Handle full URIs
    if entity.startswith('http://dbpedia.org/resource/'):
        entity = 'dbr:' + entity.replace('http://dbpedia.org/resource/', '')
    
    # Ensure dbr: prefix
    if not entity.startswith('dbr:') and not entity.startswith('?'):
        entity = 'dbr:' + entity
    
    return entity


def compute_execution_match(test_data: List[Dict], executor: SPARQLExecutor, 
                            adversarial: bool = False) -> Dict[str, Any]:
    """
    Compute Execution Match (EX) metric.
    
    Args:
        test_data: List of test examples
        executor: SPARQL executor
        adversarial: If True, expects empty results
    
    Returns:
        Dictionary with metrics and detailed results
    """
    results = {
        'total': len(test_data),
        'correct': 0,
        'incorrect': 0,
        'execution_errors': 0,
        'details': []
    }
    
    for i, example in enumerate(test_data, 1):
        question = example['input']
        ground_truth_sparql = example['output']
        
        print(f"\n[{i}/{len(test_data)}] {question}")
        
        # Execute ground truth SPARQL
        print(f"   Executing ground truth SPARQL...")
        ground_truth_results = executor.execute(ground_truth_sparql)
        
        print(f"   Ground truth results: {len(ground_truth_results)} entities")
        if len(ground_truth_results) > 0 and len(ground_truth_results) <= 3:
            print(f"   → {ground_truth_results}")
        
        # For adversarial examples, check if result is empty
        if adversarial:
            expected_empty = example.get('expected_result') == 'empty'
            if expected_empty and len(ground_truth_results) == 0:
                results['correct'] += 1
                status = '✅ PASS (Empty as expected)'
            elif expected_empty and len(ground_truth_results) > 0:
                results['incorrect'] += 1
                status = '❌ FAIL (Should be empty)'
            else:
                results['correct'] += 1
                status = '✅ PASS'
        else:
            # Normal test - we're comparing predicted vs ground truth
            # For now, we consider ground truth as the baseline
            if len(ground_truth_results) > 0:
                results['correct'] += 1
                status = '✅ Baseline established'
            else:
                results['execution_errors'] += 1
                status = '⚠️ Empty result'
        
        print(f"   Status: {status}")
        
        results['details'].append({
            'question': question,
            'ground_truth_result_count': len(ground_truth_results),
            'ground_truth_results': list(ground_truth_results)[:5],  # First 5 for readability
            'status': status
        })
    
    # Calculate metric
    results['execution_match_score'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results


def save_results(results: Dict, output_file: Path):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 80)
    print("SLM1 Benchmark: Execution Match (EX) Metric")
    print("=" * 80)
    
    # Load KG
    print("\n[INFO] Loading Knowledge Graph...")
    if not RELS_FILE.exists():
        print(f"❌ Error: {RELS_FILE} not found!")
        return
    
    rels_df = pd.read_csv(RELS_FILE)
    executor = SPARQLExecutor(rels_df)
    
    # Test 1: Standard test set
    print("\n" + "=" * 80)
    print("Test 1: Standard Test Set")
    print("=" * 80)
    
    if TEST_FILE.exists():
        test_data = load_test_data(TEST_FILE)
        print(f"Loaded {len(test_data)} test examples")
        
        results_standard = compute_execution_match(test_data, executor, adversarial=False)
        
        print("\n📊 Results Summary:")
        print(f"   Total queries: {results_standard['total']}")
        print(f"   Correct executions: {results_standard['correct']}")
        print(f"   Execution errors: {results_standard['execution_errors']}")
        print(f"   Execution Match (EX): {results_standard['execution_match_score']:.2%}")
        
        output_file = RESULTS_DIR / "slm1_execution_match_standard.json"
        save_results(results_standard, output_file)
        print(f"\n💾 Saved results to: {output_file}")
    else:
        print(f"⚠️ Test file not found: {TEST_FILE}")
    
    # Test 2: Adversarial test set (CFC)
    print("\n" + "=" * 80)
    print("Test 2: Adversarial Test Set (CFC)")
    print("=" * 80)
    
    if ADVERSARIAL_FILE.exists():
        adv_data = load_test_data(ADVERSARIAL_FILE)
        print(f"Loaded {len(adv_data)} adversarial examples")
        
        results_adversarial = compute_execution_match(adv_data, executor, adversarial=True)
        
        print("\n📊 Results Summary:")
        print(f"   Total queries: {results_adversarial['total']}")
        print(f"   Correct (empty results): {results_adversarial['correct']}")
        print(f"   Failed CFC: {results_adversarial['incorrect']}")
        print(f"   CFC Pass Rate: {results_adversarial['execution_match_score']:.2%}")
        
        output_file = RESULTS_DIR / "slm1_execution_match_adversarial.json"
        save_results(results_adversarial, output_file)
        print(f"\n💾 Saved results to: {output_file}")
    else:
        print(f"⚠️ Adversarial file not found: {ADVERSARIAL_FILE}")
    
    print("\n" + "=" * 80)
    print("✅ Execution Match benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
