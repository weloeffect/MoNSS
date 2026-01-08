#!/usr/bin/env python3
"""
SLM2 Benchmark: Hallucination Detection and Entity Coverage

This script measures the "Muzzle" constraint - ensuring SLM2 only uses
information present in the SPARQL results and doesn't hallucinate facts.

Metrics:
1. Hallucination Rate - % of outputs containing entities NOT in SPARQL results
2. Entity Coverage - % of SPARQL result entities that appear in the output
3. Abstention Correctness - % of empty results correctly handled as "I don't know"
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


# File paths
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

TEST_FILE = DATA_DIR / "train" / "slm2_test.jsonl"


def extract_entities_from_sparql_result(sparql_input: str) -> Set[str]:
    """
    Extract entities from the SPARQL Result section of the input.
    
    Example input:
    SPARQL Query:
    SELECT ?place WHERE { ... }
    
    SPARQL Result:
    - place: Melbourne
    - director: Hitchcock
    
    Returns: {'Melbourne', 'Hitchcock'}
    """
    entities = set()
    
    # Find the SPARQL Result section
    result_match = re.search(r'SPARQL Result:\s*\n(.*?)(?:\n\n|$)', sparql_input, re.DOTALL)
    if not result_match:
        return entities
    
    result_section = result_match.group(1)
    
    # Extract values after colons (e.g., "- place: Melbourne" -> "Melbourne")
    # Pattern: - <variable>: <value>
    pattern = r'-\s+\w+:\s+(.+?)(?=\n|$)'
    matches = re.findall(pattern, result_section)
    
    for match in matches:
        # Clean the entity value
        entity = match.strip()
        if entity and entity.lower() not in ['empty', 'none', 'null']:
            entities.add(entity)
    
    return entities


def extract_entities_from_output(output_text: str, kg_entities: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    Extract entities from the generated output text.
    
    Returns:
        Tuple of (matched_entities, potential_hallucinations)
        - matched_entities: entities from KG that appear in output
        - potential_hallucinations: named entities in output not from KG
    """
    output_lower = output_text.lower()
    matched_entities = set()
    
    # Check which KG entities appear in the output
    for entity in kg_entities:
        # Case-insensitive substring match
        if entity.lower() in output_lower:
            matched_entities.add(entity)
    
    # Extract potential new entities (capitalized phrases)
    # This is a heuristic - in production, use NER or exact KG matching
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    output_entities = set(re.findall(capitalized_pattern, output_text))
    
    # Remove matched KG entities from potential hallucinations
    potential_hallucinations = set()
    for output_entity in output_entities:
        # Check if this entity or any KG entity is a substring of the other
        is_from_kg = False
        for kg_entity in kg_entities:
            if (output_entity.lower() in kg_entity.lower() or 
                kg_entity.lower() in output_entity.lower()):
                is_from_kg = True
                break
        
        if not is_from_kg:
            potential_hallucinations.add(output_entity)
    
    return matched_entities, potential_hallucinations


def is_abstention(text: str) -> bool:
    """Check if the output is an abstention (I don't know)."""
    text_lower = text.lower().strip()
    
    abstention_phrases = [
        "i don't know",
        "i do not know",
        "unknown",
        "no information",
        "cannot determine",
        "not available"
    ]
    
    return any(phrase in text_lower for phrase in abstention_phrases)


def is_empty_result(sparql_input: str) -> bool:
    """Check if SPARQL result is empty."""
    result_match = re.search(r'SPARQL Result:\s*\n(.*?)(?:\n\n|$)', sparql_input, re.DOTALL)
    if not result_match:
        return True
    
    result_section = result_match.group(1).strip()
    
    # Check for empty indicators
    empty_indicators = ['empty', 'none', 'null', '']
    return (not result_section or 
            result_section.lower() in empty_indicators or
            len(result_section) < 5)


def analyze_prediction(prediction: Dict) -> Dict:
    """
    Analyze a single prediction for hallucination and entity coverage.
    
    Args:
        prediction: Dict with 'input', 'output' (ground truth), 'predicted'
        
    Returns:
        Dict with analysis results
    """
    sparql_input = prediction['input']
    ground_truth = prediction['output']
    predicted = prediction.get('predicted', '')
    
    # Extract entities from SPARQL results
    kg_entities = extract_entities_from_sparql_result(sparql_input)
    
    # Check if result is empty
    empty_result = is_empty_result(sparql_input)
    
    # Extract entities from predicted output
    matched_entities, hallucinations = extract_entities_from_output(predicted, kg_entities)
    
    # Compute entity coverage
    coverage = len(matched_entities) / len(kg_entities) if kg_entities else 1.0
    
    # Check abstention correctness
    is_empty = empty_result
    should_abstain = is_empty
    did_abstain = is_abstention(predicted)
    abstention_correct = (should_abstain == did_abstain)
    
    # Hallucination detection
    has_hallucination = len(hallucinations) > 0
    
    return {
        'kg_entities': kg_entities,
        'matched_entities': matched_entities,
        'hallucinations': hallucinations,
        'entity_coverage': coverage,
        'has_hallucination': has_hallucination,
        'is_empty_result': is_empty,
        'should_abstain': should_abstain,
        'did_abstain': did_abstain,
        'abstention_correct': abstention_correct,
        'predicted': predicted,
        'ground_truth': ground_truth
    }


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute aggregate metrics across all predictions.
    
    Returns:
        Dict with hallucination rate, avg coverage, abstention accuracy
    """
    total = len(predictions)
    
    if total == 0:
        return {
            'total': 0,
            'hallucination_rate': 0.0,
            'avg_entity_coverage': 0.0,
            'abstention_accuracy': 0.0
        }
    
    hallucination_count = 0
    coverage_scores = []
    abstention_correct_count = 0
    abstention_total = 0
    
    detailed_results = []
    
    for pred in predictions:
        analysis = analyze_prediction(pred)
        detailed_results.append(analysis)
        
        # Hallucination counting
        if analysis['has_hallucination']:
            hallucination_count += 1
        
        # Entity coverage (only for non-empty results)
        if not analysis['is_empty_result']:
            coverage_scores.append(analysis['entity_coverage'])
        
        # Abstention accuracy
        if analysis['should_abstain']:
            abstention_total += 1
            if analysis['abstention_correct']:
                abstention_correct_count += 1
    
    # Compute aggregate metrics
    hallucination_rate = hallucination_count / total
    avg_entity_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    abstention_accuracy = (abstention_correct_count / abstention_total 
                          if abstention_total > 0 else 1.0)
    
    return {
        'total': total,
        'hallucination_count': hallucination_count,
        'hallucination_rate': hallucination_rate,
        'avg_entity_coverage': avg_entity_coverage,
        'coverage_scores': coverage_scores,
        'abstention_correct': abstention_correct_count,
        'abstention_total': abstention_total,
        'abstention_accuracy': abstention_accuracy,
        'detailed_results': detailed_results
    }


def load_predictions(filepath: Path) -> List[Dict]:
    """Load predictions from JSON or JSONL file."""
    predictions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Try JSON first
        try:
            data = json.load(f)
            if isinstance(data, dict) and 'predictions' in data:
                predictions = data['predictions']
            elif isinstance(data, list):
                predictions = data
        except json.JSONDecodeError:
            # Try JSONL
            f.seek(0)
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
    
    return predictions


def save_results(results: Dict, output_file: Path):
    """Save analysis results to JSON file."""
    # Remove detailed results for summary file (too large)
    summary = {k: v for k, v in results.items() if k != 'detailed_results'}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")


def print_report(results: Dict, dataset_name: str):
    """Print a formatted report of the analysis."""
    print("\n" + "="*80)
    print(f"SLM2 Hallucination & Entity Coverage Report: {dataset_name}")
    print("="*80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Predictions: {results['total']}")
    
    print(f"\n🚨 Hallucination Metrics:")
    print(f"   Hallucination Count: {results['hallucination_count']}/{results['total']}")
    print(f"   Hallucination Rate: {results['hallucination_rate']:.2%}")
    print(f"   ✅ Target: < 5% hallucination rate")
    
    if results['hallucination_rate'] < 0.05:
        print(f"   ✅ PASS - Excellent hallucination control!")
    elif results['hallucination_rate'] < 0.10:
        print(f"   ⚠️  WARNING - Moderate hallucination detected")
    else:
        print(f"   ❌ FAIL - High hallucination rate!")
    
    print(f"\n📝 Entity Coverage:")
    print(f"   Average Coverage: {results['avg_entity_coverage']:.2%}")
    print(f"   ✅ Target: > 90% entity coverage")
    
    if results['avg_entity_coverage'] > 0.90:
        print(f"   ✅ PASS - Excellent entity coverage!")
    elif results['avg_entity_coverage'] > 0.75:
        print(f"   ⚠️  WARNING - Moderate entity coverage")
    else:
        print(f"   ❌ FAIL - Low entity coverage!")
    
    print(f"\n🛑 Abstention Correctness (\"I don't know\" on empty results):")
    print(f"   Correct Abstentions: {results['abstention_correct']}/{results['abstention_total']}")
    print(f"   Abstention Accuracy: {results['abstention_accuracy']:.2%}")
    print(f"   ✅ Target: 100% abstention accuracy")
    
    if results['abstention_accuracy'] == 1.0:
        print(f"   ✅ PASS - Perfect abstention handling!")
    elif results['abstention_accuracy'] > 0.90:
        print(f"   ⚠️  WARNING - Some abstention failures")
    else:
        print(f"   ❌ FAIL - Critical failure constraint violated!")
    
    # Show examples of hallucinations
    if results['hallucination_count'] > 0:
        print(f"\n⚠️  Hallucination Examples (first 3):")
        count = 0
        for detail in results['detailed_results']:
            if detail['has_hallucination'] and count < 3:
                print(f"\n   Example {count + 1}:")
                print(f"   KG Entities: {detail['kg_entities']}")
                print(f"   Hallucinations: {detail['hallucinations']}")
                print(f"   Output: {detail['predicted'][:100]}...")
                count += 1
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    print("🔍 SLM2 Hallucination Detection & Entity Coverage Analysis")
    print("="*80)
    
    # Check if predictions file exists
    # For now, we'll test on ground truth data (slm2_test.jsonl)
    # Later, replace with actual model predictions
    
    test_file = TEST_FILE
    
    if not test_file.exists():
        print(f"❌ Error: Test file not found: {test_file}")
        print("   Please run slm2_inference.py first to generate predictions.")
        return
    
    print(f"📂 Loading test data from: {test_file}")
    
    # Load predictions
    predictions = load_predictions(test_file)
    
    # For testing with ground truth (no 'predicted' field yet)
    # Add ground truth as predicted
    for pred in predictions:
        if 'predicted' not in pred:
            pred['predicted'] = pred['output']
    
    print(f"✅ Loaded {len(predictions)} predictions")
    
    # Compute metrics
    print("\n🔬 Analyzing predictions...")
    results = compute_metrics(predictions)
    
    # Print report
    print_report(results, "SLM2 Test Set")
    
    # Save results
    output_file = RESULTS_DIR / "slm2_hallucination_analysis.json"
    save_results(results, output_file)
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
