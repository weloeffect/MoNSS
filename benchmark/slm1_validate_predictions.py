#!/usr/bin/env python3
"""
Validate SLM1 Model Predictions for Syntactic Validity
"""

import json
import re
from pathlib import Path

# File paths - predictions in data/train/
DATA_DIR = Path("./data/train")
PREDICTIONS_STANDARD = DATA_DIR / "slm1_predictions_standard_fixed.json"
PREDICTIONS_ADVERSARIAL = DATA_DIR / "slm1_predictions_adversarial_fixed.json"

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def validate_sparql(sparql):
    """Validate SPARQL syntax using regex checks."""
    errors = []
    
    if not sparql or len(sparql.strip()) == 0:
        return {"valid": False, "error": "Empty SPARQL"}
    
    # Check 1: Has SELECT clause
    if "SELECT" not in sparql.upper():
        errors.append("Missing SELECT")
    
    # Check 2: Has WHERE clause
    if "WHERE" not in sparql.upper():
        errors.append("Missing WHERE")
    
    # Check 3: Brackets balanced
    if sparql.count("{") != sparql.count("}"):
        errors.append("Unbalanced brackets")
    
    # Check 4: Has PREFIX
    if "PREFIX" not in sparql.upper():
        errors.append("Missing PREFIX")
    
    # Check 5: Missing space after SELECT (common error)
    if re.search(r"SELECT\?", sparql):
        errors.append("Missing space after SELECT")
    
    # Check 6: Missing space before variable in predicate
    if re.search(r"dbo:\w+\?", sparql) or re.search(r"dbr:\w+\?", sparql):
        errors.append("Missing space before variable")
    
    if errors:
        return {"valid": False, "error": "; ".join(errors)}
    else:
        return {"valid": True, "error": None}


def validate_predictions(filepath, name):
    """Validate all predictions in a file."""
    print("\n" + "=" * 60)
    print("Validating: " + name)
    print("=" * 60)
    
    if not filepath.exists():
        print("File not found: " + str(filepath))
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    predictions = data.get("predictions", [])
    print("Total predictions: " + str(len(predictions)))
    
    valid_count = 0
    invalid_count = 0
    details = []
    
    for pred in predictions:
        idx = pred.get("index", 0)
        question = pred.get("question", "")
        sparql = pred.get("predicted_sparql", "")
        
        result = validate_sparql(sparql)
        
        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            print("\n[" + str(idx) + "] INVALID: " + question[:50] + "...")
            print("    Error: " + result["error"])
            print("    SPARQL: " + sparql[:80] + "...")
        
        details.append({
            "index": idx,
            "question": question,
            "valid": result["valid"],
            "error": result["error"],
            "predicted_sparql": sparql
        })
    
    total = len(predictions)
    rate = valid_count / total if total > 0 else 0
    
    print("\n" + "-" * 40)
    print("Results:")
    print("  Valid: " + str(valid_count) + "/" + str(total))
    print("  Invalid: " + str(invalid_count) + "/" + str(total))
    print("  Validity Rate: " + str(round(rate * 100, 2)) + "%")
    
    return {
        "dataset": name,
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "validity_rate": rate,
        "details": details
    }


def main():
    print("=" * 60)
    print("SLM1 Predictions Syntactic Validity Check")
    print("=" * 60)
    
    all_results = {}
    
    # Validate standard predictions
    result1 = validate_predictions(PREDICTIONS_STANDARD, "Standard Predictions")
    if result1:
        all_results["standard"] = result1
        output_file = RESULTS_DIR / "slm1_validity_predictions_standard.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result1, f, indent=2)
        print("\nSaved: " + str(output_file))
    
    # Validate adversarial predictions
    result2 = validate_predictions(PREDICTIONS_ADVERSARIAL, "Adversarial Predictions")
    if result2:
        all_results["adversarial"] = result2
        output_file = RESULTS_DIR / "slm1_validity_predictions_adversarial.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result2, f, indent=2)
        print("\nSaved: " + str(output_file))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_all = 0
    valid_all = 0
    
    for name, result in all_results.items():
        print(name.capitalize() + ": " + str(round(result["validity_rate"] * 100, 2)) + "% (" + str(result["valid"]) + "/" + str(result["total"]) + ")")
        total_all += result["total"]
        valid_all += result["valid"]
    
    if total_all > 0:
        overall = valid_all / total_all
        print("\nOverall: " + str(round(overall * 100, 2)) + "% (" + str(valid_all) + "/" + str(total_all) + ")")
        
        print("\n" + "=" * 60)
        if overall >= 0.95:
            print("PASSED: Syntactic validity >= 95%")
        else:
            print("BELOW TARGET: " + str(round(overall * 100, 2)) + "% < 95%")
            print("\nCommon issues found in your predictions:")
            print("  - 'SELECT?var' should be 'SELECT ?var'")
            print("  - 'dbo:pred?var' should be 'dbo:pred ?var'")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
