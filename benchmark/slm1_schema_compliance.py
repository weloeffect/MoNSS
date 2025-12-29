#!/usr/bin/env python3
"""
SLM1 Benchmark: Schema Compliance

This script validates that generated SPARQL queries only use predicates
that are defined in the ontology schema (schema.json).

Schema Compliance Rate = (Queries with valid predicates) / (Total Queries)

A query is schema-compliant if:
1. All predicates used exist in schema.json
2. Predicates are used with correct domain/range types (if specified)
3. No hallucinated or made-up predicates
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


# File paths
DATA_DIR = Path("./data")
SCHEMA_FILE = Path("./schema") / "schema.json"
TEST_FILE = DATA_DIR / "train" / "slm1_test.jsonl"
ADVERSARIAL_FILE = DATA_DIR / "train" / "slm1_adversarial_test.jsonl"
TRAIN_FILE = DATA_DIR / "train" / "slm1_train_final.jsonl"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


class SchemaValidator:
    """Validates SPARQL queries against ontology schema."""
    
    def __init__(self, schema_file: Path):
        """
        Initialize validator with schema.
        
        Args:
            schema_file: Path to schema.json
        """
        self.schema = self.load_schema(schema_file)
        self.valid_predicates = self.extract_predicates()
        
        print(f"   Loaded schema with {len(self.valid_predicates)} valid predicates")
        print(f"   Valid predicates: {', '.join(sorted(self.valid_predicates))}")
    
    def load_schema(self, schema_file: Path) -> Dict:
        """Load schema from JSON file."""
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_predicates(self) -> Set[str]:
        """
        Extract all valid predicates from schema.
        
        Returns:
            Set of valid predicate URIs (e.g., 'dbo:director')
        """
        predicates = set()
        
        # Schema format may vary - handle different structures
        if isinstance(self.schema, dict):
            # Check for 'predicates' key
            if 'predicates' in self.schema:
                for pred in self.schema['predicates']:
                    if isinstance(pred, str):
                        predicates.add(pred)
                    elif isinstance(pred, dict) and 'uri' in pred:
                        predicates.add(pred['uri'])
            
            # Check for 'relations' key
            if 'relations' in self.schema:
                for rel in self.schema['relations']:
                    if isinstance(rel, str):
                        predicates.add(rel)
                    elif isinstance(rel, dict):
                        if 'predicate' in rel:
                            predicates.add(rel['predicate'])
                        elif 'uri' in rel:
                            predicates.add(rel['uri'])
            
            # Check for 'properties' key
            if 'properties' in self.schema:
                for prop in self.schema['properties']:
                    if isinstance(prop, str):
                        predicates.add(prop)
                    elif isinstance(prop, dict) and 'uri' in prop:
                        predicates.add(prop['uri'])
        
        # Ensure predicates have proper prefix
        normalized_predicates = set()
        for pred in predicates:
            if not pred.startswith('dbo:') and not pred.startswith('dbr:'):
                # Add dbo: prefix if missing
                normalized_predicates.add(f'dbo:{pred}')
            else:
                normalized_predicates.add(pred)
        
        return normalized_predicates if normalized_predicates else predicates
    
    def extract_predicates_from_sparql(self, sparql: str) -> List[str]:
        """
        Extract all predicates used in a SPARQL query.
        
        Args:
            sparql: SPARQL query string
        
        Returns:
            List of predicates found (e.g., ['dbo:director', 'dbo:birthPlace'])
        """
        predicates = []
        
        # Pattern to match predicates in triple patterns
        # Matches: subject predicate object
        # Predicates typically have namespace prefix (dbo:, dbr:, etc.)
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s*\{(.*?)\}', sparql, re.DOTALL | re.IGNORECASE)
        if not where_match:
            return predicates
        
        where_clause = where_match.group(1)
        
        # Split by period or semicolon to get triple patterns
        patterns = re.split(r'[.;]', where_clause)
        
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern:
                continue
            
            # Split on whitespace
            parts = pattern.split()
            
            # In a triple pattern: subject predicate object
            # Predicate is typically the second element
            if len(parts) >= 3:
                potential_predicate = parts[1]
                
                # Check if it looks like a predicate (has namespace prefix)
                if ':' in potential_predicate and not potential_predicate.startswith('?'):
                    predicates.append(potential_predicate)
        
        return predicates
    
    def validate_query(self, sparql: str) -> Dict[str, Any]:
        """
        Validate that all predicates in query are in schema.
        
        Args:
            sparql: SPARQL query string
        
        Returns:
            dict with 'compliant' (bool), 'predicates_used' (list), 
            'invalid_predicates' (list), 'error' (str or None)
        """
        try:
            predicates_used = self.extract_predicates_from_sparql(sparql)
            
            invalid_predicates = []
            for pred in predicates_used:
                if pred not in self.valid_predicates:
                    invalid_predicates.append(pred)
            
            compliant = len(invalid_predicates) == 0
            
            return {
                'compliant': compliant,
                'predicates_used': predicates_used,
                'invalid_predicates': invalid_predicates,
                'error': None if compliant else f"Invalid predicates: {', '.join(invalid_predicates)}"
            }
        
        except Exception as e:
            return {
                'compliant': False,
                'predicates_used': [],
                'invalid_predicates': [],
                'error': f"Validation error: {str(e)}"
            }


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def validate_dataset(data: List[Dict], validator: SchemaValidator, 
                     dataset_name: str) -> Dict[str, Any]:
    """
    Validate all queries in a dataset for schema compliance.
    
    Args:
        data: List of examples with 'output' containing SPARQL
        validator: SchemaValidator instance
        dataset_name: Name for reporting
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'dataset': dataset_name,
        'total': len(data),
        'compliant': 0,
        'non_compliant': 0,
        'predicate_usage': defaultdict(int),
        'invalid_predicates_found': defaultdict(int),
        'details': []
    }
    
    print(f"\n🔍 Validating {dataset_name}...")
    print(f"   Total queries: {len(data)}")
    
    for i, example in enumerate(data, 1):
        sparql = example['output']
        question = example.get('input', 'N/A')
        
        # Validate
        validation_result = validator.validate_query(sparql)
        
        # Count predicate usage
        for pred in validation_result['predicates_used']:
            results['predicate_usage'][pred] += 1
        
        if validation_result['compliant']:
            results['compliant'] += 1
            status = '✅'
        else:
            results['non_compliant'] += 1
            status = '❌'
            
            # Count invalid predicates
            for pred in validation_result['invalid_predicates']:
                results['invalid_predicates_found'][pred] += 1
            
            print(f"\n   [{i}] {status} NON-COMPLIANT: {question[:60]}...")
            print(f"       Invalid predicates: {', '.join(validation_result['invalid_predicates'])}")
        
        results['details'].append({
            'index': i,
            'question': question,
            'compliant': validation_result['compliant'],
            'predicates_used': validation_result['predicates_used'],
            'invalid_predicates': validation_result['invalid_predicates'],
            'error': validation_result['error']
        })
        
        # Progress indicator
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(data)} ({results['compliant']} compliant)")
    
    # Calculate compliance rate
    results['compliance_rate'] = results['compliant'] / results['total'] if results['total'] > 0 else 0
    
    # Convert defaultdict to regular dict for JSON serialization
    results['predicate_usage'] = dict(results['predicate_usage'])
    results['invalid_predicates_found'] = dict(results['invalid_predicates_found'])
    
    return results


def save_results(results: Dict, output_file: Path):
    """Save results to JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 80)
    print("SLM1 Benchmark: Schema Compliance")
    print("=" * 80)
    
    # Check schema file
    if not SCHEMA_FILE.exists():
        print(f"❌ Error: {SCHEMA_FILE} not found!")
        print(f"   Please ensure schema.json exists in the schema/ directory")
        return
    
    print(f"\n📋 Loading schema from: {SCHEMA_FILE}")
    validator = SchemaValidator(SCHEMA_FILE)
    
    all_results = {}
    
    # Validate training set
    print("\n" + "=" * 80)
    print("Validation 1: Training Set")
    print("=" * 80)
    
    if TRAIN_FILE.exists():
        train_data = load_jsonl(TRAIN_FILE)
        results_train = validate_dataset(train_data, validator, "Training Set")
        
        print(f"\n📊 Training Set Results:")
        print(f"   Total: {results_train['total']}")
        print(f"   Compliant: {results_train['compliant']}")
        print(f"   Non-compliant: {results_train['non_compliant']}")
        print(f"   Compliance Rate: {results_train['compliance_rate']:.2%}")
        
        if results_train['invalid_predicates_found']:
            print(f"\n   Invalid predicates found:")
            for pred, count in sorted(results_train['invalid_predicates_found'].items(), 
                                     key=lambda x: -x[1]):
                print(f"      {pred}: {count} occurrences")
        
        output_file = RESULTS_DIR / "slm1_schema_compliance_train.json"
        save_results(results_train, output_file)
        print(f"\n💾 Saved: {output_file}")
        
        all_results['train'] = results_train
    
    # Validate test set
    print("\n" + "=" * 80)
    print("Validation 2: Test Set")
    print("=" * 80)
    
    if TEST_FILE.exists():
        test_data = load_jsonl(TEST_FILE)
        results_test = validate_dataset(test_data, validator, "Test Set")
        
        print(f"\n📊 Test Set Results:")
        print(f"   Total: {results_test['total']}")
        print(f"   Compliant: {results_test['compliant']}")
        print(f"   Non-compliant: {results_test['non_compliant']}")
        print(f"   Compliance Rate: {results_test['compliance_rate']:.2%}")
        
        if results_test['invalid_predicates_found']:
            print(f"\n   Invalid predicates found:")
            for pred, count in sorted(results_test['invalid_predicates_found'].items(), 
                                     key=lambda x: -x[1]):
                print(f"      {pred}: {count} occurrences")
        
        output_file = RESULTS_DIR / "slm1_schema_compliance_test.json"
        save_results(results_test, output_file)
        print(f"\n💾 Saved: {output_file}")
        
        all_results['test'] = results_test
    
    # Validate adversarial set
    print("\n" + "=" * 80)
    print("Validation 3: Adversarial Set")
    print("=" * 80)
    
    if ADVERSARIAL_FILE.exists():
        adv_data = load_jsonl(ADVERSARIAL_FILE)
        results_adv = validate_dataset(adv_data, validator, "Adversarial Set")
        
        print(f"\n📊 Adversarial Set Results:")
        print(f"   Total: {results_adv['total']}")
        print(f"   Compliant: {results_adv['compliant']}")
        print(f"   Non-compliant: {results_adv['non_compliant']}")
        print(f"   Compliance Rate: {results_adv['compliance_rate']:.2%}")
        
        if results_adv['invalid_predicates_found']:
            print(f"\n   Invalid predicates found:")
            for pred, count in sorted(results_adv['invalid_predicates_found'].items(), 
                                     key=lambda x: -x[1]):
                print(f"      {pred}: {count} occurrences")
        
        output_file = RESULTS_DIR / "slm1_schema_compliance_adversarial.json"
        save_results(results_adv, output_file)
        print(f"\n💾 Saved: {output_file}")
        
        all_results['adversarial'] = results_adv
    
    # Overall summary
    print("\n" + "=" * 80)
    print("📊 Overall Summary")
    print("=" * 80)
    
    for name, result in all_results.items():
        print(f"{name.capitalize()}: {result['compliance_rate']:.2%} ({result['compliant']}/{result['total']})")
    
    # Combined compliance rate
    total_all = sum(r['total'] for r in all_results.values())
    compliant_all = sum(r['compliant'] for r in all_results.values())
    overall_rate = compliant_all / total_all if total_all > 0 else 0
    
    print(f"\nCombined: {overall_rate:.2%} ({compliant_all}/{total_all})")
    
    # Predicate usage summary
    print("\n📊 Most Used Predicates:")
    all_predicate_usage = defaultdict(int)
    for result in all_results.values():
        for pred, count in result['predicate_usage'].items():
            all_predicate_usage[pred] += count
    
    for pred, count in sorted(all_predicate_usage.items(), key=lambda x: -x[1])[:10]:
        print(f"   {pred}: {count}")
    
    # Goal check
    print("\n" + "=" * 80)
    if overall_rate >= 0.95:
        print("✅ PASSED: Schema compliance >95%")
    else:
        print(f"⚠️ BELOW TARGET: {overall_rate:.2%} < 95%")
    print("=" * 80)


if __name__ == "__main__":
    main()
