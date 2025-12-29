#!/usr/bin/env python3
"""
SLM1 Benchmark: Syntactic Validity

This script validates that generated SPARQL queries are syntactically correct
and can be parsed by a SPARQL parser (rdflib).

Syntactic Validity Rate = (Valid SPARQL) / (Total Queries)

A query is syntactically valid if:
1. It follows SPARQL grammar rules
2. It can be parsed without errors
3. PREFIX declarations are correct
4. Brackets, quotes, and punctuation are balanced
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Check if rdflib is installed
try:
    from rdflib.plugins.sparql import prepareQuery
    from rdflib.plugins.sparql.parser import ParseException
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    print("⚠️ Warning: rdflib not installed. Install with: pip install rdflib")


# File paths
DATA_DIR = Path("./data")
TEST_FILE = DATA_DIR / "train" / "slm1_test.jsonl"
ADVERSARIAL_FILE = DATA_DIR / "train" / "slm1_adversarial_test.jsonl"
TRAIN_FILE = DATA_DIR / "train" / "slm1_train_final.jsonl"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


class SPARQLSyntaxValidator:
    """Validates SPARQL query syntax."""
    
    def __init__(self, use_rdflib: bool = True):
        """
        Initialize validator.
        
        Args:
            use_rdflib: If True, use rdflib parser (strict). If False, use regex-based validation.
        """
        self.use_rdflib = use_rdflib and RDFLIB_AVAILABLE
        
        if self.use_rdflib:
            print("   Using rdflib for strict SPARQL validation")
        else:
            print("   Using regex-based validation (less strict)")
    
    def validate_with_rdflib(self, sparql: str) -> Dict[str, Any]:
        """
        Validate SPARQL using rdflib parser.
        
        Returns:
            dict with 'valid' (bool), 'error' (str or None)
        """
        try:
            # Attempt to parse the query
            prepareQuery(sparql)
            return {'valid': True, 'error': None}
        except ParseException as e:
            return {'valid': False, 'error': f"ParseException: {str(e)}"}
        except Exception as e:
            return {'valid': False, 'error': f"Exception: {str(e)}"}
    
    def validate_with_regex(self, sparql: str) -> Dict[str, Any]:
        """
        Basic regex-based validation for SPARQL structure.
        Less strict than rdflib, but doesn't require external library.
        
        Returns:
            dict with 'valid' (bool), 'error' (str or None)
        """
        errors = []
        
        # Check 1: Has SELECT clause
        if 'SELECT' not in sparql.upper():
            errors.append("Missing SELECT clause")
        
        # Check 2: Has WHERE clause
        if 'WHERE' not in sparql.upper():
            errors.append("Missing WHERE clause")
        
        # Check 3: Brackets are balanced
        if sparql.count('{') != sparql.count('}'):
            errors.append("Unbalanced brackets")
        
        # Check 4: Has PREFIX declarations (for DBpedia queries)
        if 'PREFIX' not in sparql.upper():
            errors.append("Missing PREFIX declarations")
        
        # Check 5: Check for common syntax errors
        if sparql.count('(') != sparql.count(')'):
            errors.append("Unbalanced parentheses")
        
        # Check 6: Variables start with ?
        import re
        vars_in_select = re.findall(r'SELECT\s+([^\s{]+)', sparql, re.IGNORECASE)
        for var in vars_in_select:
            if not var.startswith('?') and var.upper() not in ['DISTINCT', 'REDUCED']:
                errors.append(f"Invalid variable in SELECT: {var}")
        
        if errors:
            return {'valid': False, 'error': '; '.join(errors)}
        else:
            return {'valid': True, 'error': None}
    
    def validate(self, sparql: str) -> Dict[str, Any]:
        """
        Main validation method.
        
        Returns:
            dict with 'valid' (bool), 'error' (str or None), 'method' (str)
        """
        if self.use_rdflib:
            result = self.validate_with_rdflib(sparql)
            result['method'] = 'rdflib'
        else:
            result = self.validate_with_regex(sparql)
            result['method'] = 'regex'
        
        return result


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def validate_dataset(data: List[Dict], validator: SPARQLSyntaxValidator, 
                     dataset_name: str) -> Dict[str, Any]:
    """
    Validate all SPARQL queries in a dataset.
    
    Args:
        data: List of examples with 'output' containing SPARQL
        validator: SPARQLSyntaxValidator instance
        dataset_name: Name for reporting
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'dataset': dataset_name,
        'total': len(data),
        'valid': 0,
        'invalid': 0,
        'details': []
    }
    
    print(f"\n🔍 Validating {dataset_name}...")
    print(f"   Total queries: {len(data)}")
    
    for i, example in enumerate(data, 1):
        sparql = example['output']
        question = example.get('input', 'N/A')
        
        # Validate
        validation_result = validator.validate(sparql)
        
        if validation_result['valid']:
            results['valid'] += 1
            status = '✅'
        else:
            results['invalid'] += 1
            status = '❌'
            print(f"\n   [{i}] {status} INVALID: {question[:60]}...")
            print(f"       Error: {validation_result['error']}")
        
        results['details'].append({
            'index': i,
            'question': question,
            'valid': validation_result['valid'],
            'error': validation_result['error'],
            'validation_method': validation_result['method'],
            'sparql': sparql[:200] + '...' if len(sparql) > 200 else sparql
        })
        
        # Progress indicator
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(data)} ({results['valid']} valid, {results['invalid']} invalid)")
    
    # Calculate rate
    results['validity_rate'] = results['valid'] / results['total'] if results['total'] > 0 else 0
    
    return results


def save_results(results: Dict, output_file: Path):
    """Save results to JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 80)
    print("SLM1 Benchmark: Syntactic Validity")
    print("=" * 80)
    
    if not RDFLIB_AVAILABLE:
        print("\n⚠️ rdflib not available. Using basic regex validation.")
        print("   For strict validation, install: pip install rdflib\n")
        use_rdflib = False
    else:
        use_rdflib = True
    
    validator = SPARQLSyntaxValidator(use_rdflib=use_rdflib)
    
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
        print(f"   Valid: {results_train['valid']}")
        print(f"   Invalid: {results_train['invalid']}")
        print(f"   Validity Rate: {results_train['validity_rate']:.2%}")
        
        output_file = RESULTS_DIR / "slm1_syntactic_validity_train.json"
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
        print(f"   Valid: {results_test['valid']}")
        print(f"   Invalid: {results_test['invalid']}")
        print(f"   Validity Rate: {results_test['validity_rate']:.2%}")
        
        output_file = RESULTS_DIR / "slm1_syntactic_validity_test.json"
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
        print(f"   Valid: {results_adv['valid']}")
        print(f"   Invalid: {results_adv['invalid']}")
        print(f"   Validity Rate: {results_adv['validity_rate']:.2%}")
        
        output_file = RESULTS_DIR / "slm1_syntactic_validity_adversarial.json"
        save_results(results_adv, output_file)
        print(f"\n💾 Saved: {output_file}")
        
        all_results['adversarial'] = results_adv
    
    # Overall summary
    print("\n" + "=" * 80)
    print("📊 Overall Summary")
    print("=" * 80)
    
    for name, result in all_results.items():
        print(f"{name.capitalize()}: {result['validity_rate']:.2%} ({result['valid']}/{result['total']})")
    
    # Combined validity rate
    total_all = sum(r['total'] for r in all_results.values())
    valid_all = sum(r['valid'] for r in all_results.values())
    overall_rate = valid_all / total_all if total_all > 0 else 0
    
    print(f"\nCombined: {overall_rate:.2%} ({valid_all}/{total_all})")
    
    # Goal check
    print("\n" + "=" * 80)
    if overall_rate >= 0.95:
        print("✅ PASSED: Syntactic validity >95%")
    else:
        print(f"⚠️ BELOW TARGET: {overall_rate:.2%} < 95%")
    print("=" * 80)


if __name__ == "__main__":
    main()
