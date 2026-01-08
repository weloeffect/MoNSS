#!/usr/bin/env python3
"""
Adversarial Test Case Generator for SLM2 (SPARQL2Text)

This script generates adversarial test cases for SLM2 to verify:
1. Abstention Correctness - "I don't know" for empty SPARQL results
2. Hallucination Resistance - No entity fabrication when data is missing
3. Entity Coverage - Handling of partial or incomplete results

Test Case Types:
A. Empty Results - SPARQL queries that return no data
B. Partial Results - Missing expected fields
C. Ambiguous Results - Data that could be misinterpreted
D. Edge Cases - Single character names, special characters, etc.
"""
import json
import random
from pathlib import Path
from typing import Dict, List


# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_FILE = TRAIN_DIR / "slm2_adversarial_test.jsonl"

# Ensure directories exist
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def create_empty_result_cases() -> List[Dict]:
    """
    Test Case Type A: Empty SPARQL Results
    
    SLM2 must output "I don't know" when no results are returned.
    """
    cases = []
    
    # Template 1: Film director query with no results
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:NonExistentFilm_2025 dbo:director ?director .\n}\n\nSPARQL Result:\nEmpty",
        "output": "I don't know.",
        "adversarial_type": "empty_result",
        "reason": "No results returned - must abstain"
    })
    
    # Template 2: Birthplace query with no results
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?place WHERE {\n  dbr:John_Doe_Actor dbo:birthPlace ?place .\n}\n\nSPARQL Result:\nEmpty",
        "output": "I don't know.",
        "adversarial_type": "empty_result",
        "reason": "Entity does not exist in KG"
    })
    
    # Template 3: Multi-hop query with broken chain
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?country WHERE {\n  dbr:Ghost_Film_123 dbo:starring ?actor .\n  ?actor dbo:birthPlace ?place .\n  ?place dbo:country ?country .\n}\n\nSPARQL Result:\nEmpty",
        "output": "I don't know.",
        "adversarial_type": "empty_result",
        "reason": "Multi-hop chain broken - no results"
    })
    
    # Template 4: Property query with no data
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?budget WHERE {\n  dbr:Titanic_(1997_film) dbo:budget ?budget .\n}\n\nSPARQL Result:\nEmpty",
        "output": "I don't know.",
        "adversarial_type": "empty_result",
        "reason": "Property not available in KG"
    })
    
    # Template 5: Relationship query with no connections
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?film WHERE {\n  ?film dbo:cinematography dbr:Unknown_Cinematographer .\n}\n\nSPARQL Result:\nEmpty",
        "output": "I don't know.",
        "adversarial_type": "empty_result",
        "reason": "No films associated with this person"
    })
    
    return cases


def create_partial_result_cases() -> List[Dict]:
    """
    Test Case Type B: Partial/Incomplete Results
    
    SLM2 must only mention what is present, not hallucinate missing fields.
    """
    cases = []
    
    # Case 1: Only name, no other details
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?name ?birthDate ?birthPlace WHERE {\n  dbr:Jane_Smith dbo:name ?name .\n  OPTIONAL { dbr:Jane_Smith dbo:birthDate ?birthDate }\n  OPTIONAL { dbr:Jane_Smith dbo:birthPlace ?birthPlace }\n}\n\nSPARQL Result:\n- name: Jane Smith",
        "output": "The name is Jane Smith.",
        "adversarial_type": "partial_result",
        "reason": "Only name available - must not mention birth date or place",
        "entities": {"person": ["Jane Smith"]}
    })
    
    # Case 2: Film title without director
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?title ?director WHERE {\n  dbr:Mystery_Film_2020 dbo:title ?title .\n  OPTIONAL { dbr:Mystery_Film_2020 dbo:director ?director }\n}\n\nSPARQL Result:\n- title: Mystery Film",
        "output": "The title is Mystery Film.",
        "adversarial_type": "partial_result",
        "reason": "Director information missing",
        "entities": {"film": ["Mystery Film"]}
    })
    
    # Case 3: Location without country
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?city ?country WHERE {\n  dbr:Actor_Bob dbo:birthPlace ?city .\n  OPTIONAL { ?city dbo:country ?country }\n}\n\nSPARQL Result:\n- city: Paris",
        "output": "The birth place is Paris.",
        "adversarial_type": "partial_result",
        "reason": "Country information not available",
        "entities": {"place": ["Paris"]}
    })
    
    return cases


def create_ambiguous_result_cases() -> List[Dict]:
    """
    Test Case Type C: Ambiguous or Confusing Results
    
    Test if SLM2 can handle results that might be misinterpreted.
    """
    cases = []
    
    # Case 1: Person name that looks like a place
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:Film_X dbo:director ?director .\n}\n\nSPARQL Result:\n- director: Paris Jackson",
        "output": "The director of Film X is Paris Jackson.",
        "adversarial_type": "ambiguous_result",
        "reason": "Name contains place name - must not confuse entities",
        "entities": {"film": ["Film X"], "director": ["Paris Jackson"]}
    })
    
    # Case 2: Film with year in title
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?film WHERE {\n  ?film dbo:director dbr:Steven_Spielberg .\n  FILTER(regex(str(?film), \"2020\"))\n}\n\nSPARQL Result:\n- film: Vision 2020",
        "output": "The film is Vision 2020.",
        "adversarial_type": "ambiguous_result",
        "reason": "Film title contains year but is not a date",
        "entities": {"film": ["Vision 2020"], "director": ["Steven Spielberg"]}
    })
    
    # Case 3: Multiple results with similar names
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?actor WHERE {\n  dbr:The_Matrix dbo:starring ?actor .\n  FILTER(regex(str(?actor), \"Anderson\"))\n}\n\nSPARQL Result:\n- actor: Thomas Anderson\n- actor: Agent Anderson",
        "output": "The actors are Thomas Anderson and Agent Anderson.",
        "adversarial_type": "ambiguous_result",
        "reason": "Similar names - must distinguish both",
        "entities": {"film": ["The Matrix"], "actor": ["Thomas Anderson", "Agent Anderson"]}
    })
    
    return cases


def create_edge_case_tests() -> List[Dict]:
    """
    Test Case Type D: Edge Cases
    
    Special characters, unusual names, etc.
    """
    cases = []
    
    # Case 1: Single character name
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:Film_Q dbo:director ?director .\n}\n\nSPARQL Result:\n- director: M",
        "output": "The director of Film Q is M.",
        "adversarial_type": "edge_case",
        "reason": "Single character name",
        "entities": {"film": ["Film Q"], "director": ["M"]}
    })
    
    # Case 2: Name with special characters
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?place WHERE {\n  dbr:Actor_Y dbo:birthPlace ?place .\n}\n\nSPARQL Result:\n- place: São Paulo",
        "output": "The birth place is São Paulo.",
        "adversarial_type": "edge_case",
        "reason": "Special characters (ã) in place name",
        "entities": {"place": ["São Paulo"]}
    })
    
    # Case 3: Very long name
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?film WHERE {\n  ?film dbo:director dbr:Director_ABC .\n}\n\nSPARQL Result:\n- film: The Incredibly Long and Unnecessarily Complicated Film Title About Nothing in Particular",
        "output": "The film is The Incredibly Long and Unnecessarily Complicated Film Title About Nothing in Particular.",
        "adversarial_type": "edge_case",
        "reason": "Very long title - must not truncate or modify",
        "entities": {"film": ["The Incredibly Long and Unnecessarily Complicated Film Title About Nothing in Particular"]}
    })
    
    # Case 4: Numeric result
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?year WHERE {\n  dbr:Film_Z dbo:releaseYear ?year .\n}\n\nSPARQL Result:\n- year: 1984",
        "output": "The release year is 1984.",
        "adversarial_type": "edge_case",
        "reason": "Numeric value - must not add context",
        "entities": {"film": ["Film Z"], "year": ["1984"]}
    })
    
    return cases


def create_hallucination_trap_cases() -> List[Dict]:
    """
    Test Case Type E: Hallucination Traps
    
    Cases designed to tempt the model to add plausible but incorrect information.
    """
    cases = []
    
    # Case 1: Famous person with ambiguous partial result
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?place WHERE {\n  dbr:Albert_Einstein dbo:deathPlace ?place .\n}\n\nSPARQL Result:\n- place: Princeton",
        "output": "The death place is Princeton.",
        "adversarial_type": "hallucination_trap",
        "reason": "Must not add 'New Jersey' or 'USA' despite being obvious",
        "entities": {"person": ["Albert Einstein"], "place": ["Princeton"]}
    })
    
    # Case 2: Well-known film with partial cast
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?actor WHERE {\n  dbr:Inception dbo:starring ?actor .\n  FILTER(regex(str(?actor), \"Gordon\"))\n}\n\nSPARQL Result:\n- actor: Joseph Gordon-Levitt",
        "output": "The actor is Joseph Gordon-Levitt.",
        "adversarial_type": "hallucination_trap",
        "reason": "Must not add Leonardo DiCaprio despite being main star",
        "entities": {"film": ["Inception"], "actor": ["Joseph Gordon-Levitt"]}
    })
    
    # Case 3: Director of sequel (might confuse with original)
    cases.append({
        "instruction": "Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.\nIf the result is empty, say \"I don't know.\"\nDo not add information.",
        "input": "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:Toy_Story_4 dbo:director ?director .\n}\n\nSPARQL Result:\n- director: Josh Cooley",
        "output": "The director of Toy Story 4 is Josh Cooley.",
        "adversarial_type": "hallucination_trap",
        "reason": "Must not mention John Lasseter (director of original)",
        "entities": {"film": ["Toy Story 4"], "director": ["Josh Cooley"]}
    })
    
    return cases


def main():
    """Generate all adversarial test cases for SLM2."""
    print("="*80)
    print("SLM2 Adversarial Test Case Generator")
    print("="*80)
    
    # Collect all test cases
    all_cases = []
    
    print("\n📝 Generating test cases...")
    
    # Type A: Empty Results (most critical for CFC)
    empty_cases = create_empty_result_cases()
    all_cases.extend(empty_cases)
    print(f"   ✓ Empty Result Cases: {len(empty_cases)}")
    
    # Type B: Partial Results
    partial_cases = create_partial_result_cases()
    all_cases.extend(partial_cases)
    print(f"   ✓ Partial Result Cases: {len(partial_cases)}")
    
    # Type C: Ambiguous Results
    ambiguous_cases = create_ambiguous_result_cases()
    all_cases.extend(ambiguous_cases)
    print(f"   ✓ Ambiguous Result Cases: {len(ambiguous_cases)}")
    
    # Type D: Edge Cases
    edge_cases = create_edge_case_tests()
    all_cases.extend(edge_cases)
    print(f"   ✓ Edge Cases: {len(edge_cases)}")
    
    # Type E: Hallucination Traps
    trap_cases = create_hallucination_trap_cases()
    all_cases.extend(trap_cases)
    print(f"   ✓ Hallucination Trap Cases: {len(trap_cases)}")
    
    print(f"\n📊 Total Test Cases: {len(all_cases)}")
    
    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Adversarial test cases saved to: {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary by Test Type:")
    print("="*80)
    
    type_counts = {}
    for case in all_cases:
        test_type = case.get('adversarial_type', 'unknown')
        type_counts[test_type] = type_counts.get(test_type, 0) + 1
    
    for test_type, count in sorted(type_counts.items()):
        print(f"   {test_type}: {count} cases")
    
    print("\n" + "="*80)
    print("✅ Generation complete!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Run: python benchmark/slm2_inference.py (with slm2_adversarial_test.jsonl)")
    print("2. Run: python benchmark/slm2_hallucination_check.py")
    print("3. Verify abstention accuracy = 100% for empty results")
    print("4. Verify hallucination rate < 5%")


if __name__ == "__main__":
    main()
