#!/usr/bin/env python3
"""
Adversarial Query Generator for SLM1 CFC (Critical Failure Constraint) Testing

Generates "zero-result" queries that test if the system correctly returns
"I don't know" or empty results for non-existent relationships in the KG.

This script:
1. Loads entities from nodes.csv
2. Loads predicates from schema.json
3. Checks relationships.csv for what exists
4. Generates adversarial queries for non-existent relationships
"""

import json
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict

# Set random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# File paths
DATA_DIR = Path("./data")
NODES_FILE = DATA_DIR / "processed" / "nodes.csv"
RELS_FILE = DATA_DIR / "processed" / "relationships.csv"
SCHEMA_FILE = Path("./schema") / "schema.json"
OUTPUT_FILE = DATA_DIR / "train" / "slm1_adversarial_test.jsonl"

# Target number of adversarial examples
TARGET_ADVERSARIAL = 50


def load_knowledge_graph():
    """Load nodes and relationships from CSV files."""
    print("📊 Loading Knowledge Graph...")
    
    nodes_df = pd.read_csv(NODES_FILE)
    rels_df = pd.read_csv(RELS_FILE)
    
    print(f"   ✅ Loaded {len(nodes_df)} nodes")
    print(f"   ✅ Loaded {len(rels_df)} relationships")
    
    return nodes_df, rels_df


def load_schema():
    """Load schema to get available predicates."""
    print("\n📋 Loading Schema...")
    
    with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    # Extract predicates from schema
    predicates = []
    if isinstance(schema, dict):
        if 'predicates' in schema:
            predicates = schema['predicates']
        elif 'relations' in schema:
            predicates = schema['relations']
    
    print(f"   ✅ Found {len(predicates)} predicates in schema")
    
    return predicates, schema


def build_relationship_index(rels_df):
    """
    Build an index of what relationships exist in the KG.
    Returns: dict mapping (subject, predicate) -> [objects]
    """
    relationship_map = defaultdict(set)
    
    for _, row in rels_df.iterrows():
        subject = row['subject']
        predicate = row['predicate']
        obj = row['object']
        relationship_map[(subject, predicate)].add(obj)
    
    return relationship_map


def get_entities_by_type(nodes_df):
    """Group entities by their type."""
    entity_groups = defaultdict(list)
    
    for _, row in nodes_df.iterrows():
        entity_id = row['id']
        entity_type = row.get('type', 'Unknown')
        entity_groups[entity_type].append(entity_id)
    
    return entity_groups


def generate_adversarial_queries(nodes_df, rels_df, predicates, target_count=50):
    """
    Generate adversarial queries for non-existent relationships.
    
    Strategy:
    1. Find valid entities
    2. For each entity, try predicates that DON'T exist in the KG
    3. Create SPARQL queries that should return empty results
    """
    print(f"\n🎯 Generating {target_count} adversarial queries...")
    
    relationship_map = build_relationship_index(rels_df)
    entity_groups = get_entities_by_type(nodes_df)
    
    adversarial_examples = []
    
    # Common predicate mappings for question generation
    predicate_to_question = {
        'dbo:director': ('director', 'Who directed {entity}?'),
        'dbo:starring': ('starring actor', 'Who starred in {entity}?'),
        'dbo:author': ('author', 'Who wrote {entity}?'),
        'dbo:producer': ('producer', 'Who produced {entity}?'),
        'dbo:writer': ('writer', 'Who was the writer of {entity}?'),
        'dbo:birthPlace': ('birthplace', 'Where was {entity} born?'),
        'dbo:spouse': ('spouse', 'Who is the spouse of {entity}?'),
        'dbo:occupation': ('occupation', 'What is the occupation of {entity}?'),
        'dbo:award': ('award', 'What awards did {entity} receive?'),
        'dbo:nationality': ('nationality', 'What is the nationality of {entity}?'),
    }
    
    # Sample entities
    all_entities = nodes_df['id'].tolist()
    random.shuffle(all_entities)
    
    attempts = 0
    max_attempts = target_count * 10  # Prevent infinite loop
    
    while len(adversarial_examples) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Pick a random entity
        entity = random.choice(all_entities)
        
        # Pick a random predicate
        predicate = random.choice(list(predicate_to_question.keys()))
        
        # Check if this relationship DOES NOT exist
        if (entity, predicate) not in relationship_map:
            # This is a valid adversarial example!
            
            # Clean entity name for question
            entity_name = entity.replace('dbr:', '').replace('_', ' ')
            
            # Generate question
            _, question_template = predicate_to_question[predicate]
            question = question_template.format(entity=entity_name)
            
            # Determine schema context based on predicate
            if 'director' in predicate or 'starring' in predicate:
                context = "Schema: Film --director--> Person"
            elif 'author' in predicate or 'writer' in predicate:
                context = "Schema: Book --author--> Person"
            elif 'birthPlace' in predicate:
                context = "Schema: Person --birthPlace--> Place"
            else:
                context = "Schema: Entity --relation--> Value"
            
            # Generate SPARQL that SHOULD return no results
            sparql = f"""PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
SELECT ?answer WHERE {{
  {entity} {predicate} ?answer .
}}"""
            
            example = {
                "instruction": "Translate the question into a SPARQL query.",
                "input": question,
                "context": context,
                "output": sparql,
                "expected_result": "empty",  # Special marker for adversarial
                "adversarial": True,
                "reason": f"Relationship ({entity}, {predicate}) does not exist in KG"
            }
            
            adversarial_examples.append(example)
    
    print(f"   ✅ Generated {len(adversarial_examples)} adversarial examples")
    print(f"   📊 Success rate: {len(adversarial_examples)/attempts*100:.1f}%")
    
    return adversarial_examples


def save_jsonl(data, filepath):
    """Save list of dictionaries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    print("=" * 80)
    print("SLM1 Adversarial Query Generation (CFC Testing)")
    print("=" * 80)
    
    # Check files exist
    if not NODES_FILE.exists():
        print(f"❌ Error: {NODES_FILE} not found!")
        print(f"   Run build_nodes.py first to generate the node data.")
        return
    
    if not RELS_FILE.exists():
        print(f"❌ Error: {RELS_FILE} not found!")
        print(f"   Run build_relationships.py first to generate relationship data.")
        return
    
    if not SCHEMA_FILE.exists():
        print(f"❌ Error: {SCHEMA_FILE} not found!")
        return
    
    # Load data
    nodes_df, rels_df = load_knowledge_graph()
    predicates, schema = load_schema()
    
    # Generate adversarial queries
    adversarial_examples = generate_adversarial_queries(
        nodes_df, rels_df, predicates, target_count=TARGET_ADVERSARIAL
    )
    
    # Save to file
    print(f"\n💾 Saving adversarial test set...")
    save_jsonl(adversarial_examples, OUTPUT_FILE)
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 80)
    print("✅ Adversarial query generation complete!")
    print("=" * 80)
    print(f"\nGenerated {len(adversarial_examples)} adversarial queries")
    print(f"\nThese queries test the Critical Failure Constraint (CFC):")
    print(f"• System MUST return empty results or 'I don't know'")
    print(f"• System MUST NOT hallucinate or make up answers")
    print(f"• All queries ask for relationships that DON'T exist in the KG")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
