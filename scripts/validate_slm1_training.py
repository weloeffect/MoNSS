#!/usr/bin/env python3
"""
Validate SLM1 Training Data for Syntactic Issues
"""

import json
import re
from pathlib import Path

DATA_DIR = Path("./data/train")
TRAIN_FILE = DATA_DIR / "slm1_train_final.jsonl"


def validate_sparql(sparql):
    """Check SPARQL for common issues."""
    errors = []
    
    if not sparql:
        return {"valid": False, "error": "Empty SPARQL"}
    
    # Check for missing space after SELECT
    if re.search(r"SELECT\?", sparql):
        errors.append("Missing space after SELECT")
    
    # Check for missing space before variable in predicate
    if re.search(r"dbo:\w+\?", sparql) or re.search(r"dbr:\w+\?", sparql):
        errors.append("Missing space before variable")
    
    # Check for missing space after period
    if re.search(r"\.\?", sparql):
        errors.append("Missing space after period")
    
    # Check basic structure
    if "SELECT" not in sparql.upper():
        errors.append("Missing SELECT")
    if "WHERE" not in sparql.upper():
        errors.append("Missing WHERE")
    if "PREFIX" not in sparql.upper():
        errors.append("Missing PREFIX")
    
    # Check brackets
    if sparql.count("{") != sparql.count("}"):
        errors.append("Unbalanced brackets")
    
    if errors:
        return {"valid": False, "error": "; ".join(errors)}
    return {"valid": True, "error": None}


def main():
    print("=" * 60)
    print("Validating SLM1 Training Data")
    print("=" * 60)
    
    if not TRAIN_FILE.exists():
        print("File not found: " + str(TRAIN_FILE))
        return
    
    data = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print("Total samples: " + str(len(data)))
    
    valid_count = 0
    invalid_count = 0
    issues = []
    
    for i, item in enumerate(data, 1):
        sparql = item.get("output", "")
        question = item.get("input", "")
        
        result = validate_sparql(sparql)
        
        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            issues.append({
                "index": i,
                "question": question,
                "error": result["error"],
                "sparql": sparql[:100]
            })
    
    print("\n" + "-" * 40)
    print("Results:")
    print("  Valid: " + str(valid_count) + "/" + str(len(data)))
    print("  Invalid: " + str(invalid_count) + "/" + str(len(data)))
    print("  Validity Rate: " + str(round(valid_count / len(data) * 100, 2)) + "%")
    
    if issues:
        print("\n" + "-" * 40)
        print("Issues found:")
        for issue in issues[:10]:  # Show first 10
            print("\n[" + str(issue["index"]) + "] " + issue["question"][:50] + "...")
            print("    Error: " + issue["error"])
            print("    SPARQL: " + issue["sparql"] + "...")
    else:
        print("\n✓ No issues found in training data!")
        print("\nThe training data has correct formatting.")
        print("The model is not learning to preserve spaces during generation.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
