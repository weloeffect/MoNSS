#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLM1 Master Benchmark Runner

This script runs all three SLM1 benchmarks in sequence:
1. Syntactic Validity - Are SPARQL queries syntactically correct?
2. Schema Compliance - Do queries use valid predicates from schema?
3. Execution Match - Do queries return correct results from KG?

Usage:
    python benchmark/slm1_benchmark_runner.py
    
Or run individual benchmarks:
    python benchmark/slm1_syntactic_validity.py
    python benchmark/slm1_schema_compliance.py
    python benchmark/slm1_execution_match.py
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess
from pathlib import Path
import json
from datetime import datetime


BENCHMARK_DIR = Path("./benchmark")
RESULTS_DIR = Path("./results")


def run_benchmark(script_name: str, description: str) -> bool:
    """
    Run a benchmark script and return success status.
    
    Args:
        script_name: Name of the script file
        description: Description for display
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    
    script_path = BENCHMARK_DIR / script_name
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False


def summarize_results():
    """Load and summarize all benchmark results."""
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Syntactic Validity
    print("\n1. Syntactic Validity")
    print("-" * 40)
    syntax_files = [
        'slm1_syntactic_validity_train.json',
        'slm1_syntactic_validity_test.json',
        'slm1_syntactic_validity_adversarial.json'
    ]
    
    syntax_total = 0
    syntax_valid = 0
    
    for filename in syntax_files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                dataset = data['dataset']
                rate = data['validity_rate']
                total = data['total']
                valid = data['valid']
                
                print(f"   {dataset}: {rate:.2%} ({valid}/{total})")
                syntax_total += total
                syntax_valid += valid
    
    syntax_rate = syntax_valid / syntax_total if syntax_total > 0 else 0
    print(f"   Overall: {syntax_rate:.2%} ({syntax_valid}/{syntax_total})")
    
    summary['benchmarks']['syntactic_validity'] = {
        'rate': syntax_rate,
        'valid': syntax_valid,
        'total': syntax_total,
        'passed': syntax_rate >= 0.95
    }
    
    # Schema Compliance
    print("\n2. Schema Compliance")
    print("-" * 40)
    schema_files = [
        'slm1_schema_compliance_train.json',
        'slm1_schema_compliance_test.json',
        'slm1_schema_compliance_adversarial.json'
    ]
    
    schema_total = 0
    schema_compliant = 0
    
    for filename in schema_files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                dataset = data['dataset']
                rate = data['compliance_rate']
                total = data['total']
                compliant = data['compliant']
                
                print(f"   {dataset}: {rate:.2%} ({compliant}/{total})")
                schema_total += total
                schema_compliant += compliant
    
    schema_rate = schema_compliant / schema_total if schema_total > 0 else 0
    print(f"   Overall: {schema_rate:.2%} ({schema_compliant}/{schema_total})")
    
    summary['benchmarks']['schema_compliance'] = {
        'rate': schema_rate,
        'compliant': schema_compliant,
        'total': schema_total,
        'passed': schema_rate >= 0.95
    }
    
    # Execution Match
    print("\n3. Execution Match (EX)")
    print("-" * 40)
    exec_files = [
        'slm1_execution_match_standard.json',
        'slm1_execution_match_adversarial.json'
    ]
    
    for filename in exec_files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                rate = data.get('execution_match_score', 0)
                correct = data.get('correct', 0)
                total = data.get('total', 0)
                
                test_type = 'Standard' if 'standard' in filename else 'Adversarial (CFC)'
                print(f"   {test_type}: {rate:.2%} ({correct}/{total})")
        else:
            print(f"   {filename}: Not available")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    all_passed = True
    
    for benchmark, results in summary['benchmarks'].items():
        status = "[PASS]" if results['passed'] else "[FAIL]"
        print(f"{status} {benchmark.replace('_', ' ').title()}: {results['rate']:.2%}")
        if not results['passed']:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] All benchmarks passed the 95% threshold!")
    else:
        print("[NEEDS WORK] Some benchmarks below 95% threshold")
    print("=" * 80)
    
    # Save summary
    summary_file = RESULTS_DIR / "slm1_benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SAVED] Summary saved to: {summary_file}")
    
    return all_passed


def main():
    print("=" * 80)
    print("SLM1 MASTER BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Run benchmarks
    benchmarks = [
        ("slm1_syntactic_validity.py", "Syntactic Validity Check"),
        ("slm1_schema_compliance.py", "Schema Compliance Check"),
        ("slm1_execution_match.py", "Execution Match (EX) Metric")
    ]
    
    results = {}
    for script, description in benchmarks:
        success = run_benchmark(script, description)
        results[script] = success
        
        if not success:
            print(f"\n[WARNING] {description} encountered issues")
    
    # Summarize all results
    all_passed = summarize_results()
    
    # Exit with appropriate code
    if all_passed:
        print("\n[SUCCESS] Benchmark complete!")
        return 0
    else:
        print("\n[WARNING] Benchmark complete with warnings")
        return 1


if __name__ == "__main__":
    sys.exit(main())
