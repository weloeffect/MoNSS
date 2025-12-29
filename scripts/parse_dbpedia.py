from rdflib import Graph
import pandas as pd
from tqdm import tqdm
import os

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = "data/raw/dbpedia"
OUT_DIR = "data/processed"

INSTANCE_TYPES_FILE = os.path.join(RAW_DIR, "instance_types_en.ttl")
RELATIONS_FILE = os.path.join(RAW_DIR, "mappingbased_objects_en.ttl")

# -----------------------------
# Schema filters
# -----------------------------
ALLOWED_TYPES = {
    "Person", "Film", "Book", "Place", "Organisation",
    "School", "Scientist", "Politician"
}

ALLOWED_RELATIONS = {
    "director", "author", "starring", "writer",
    "country", "language", "birthPlace", "producer"
}

# -----------------------------
# Helper functions
# -----------------------------
def clean_uri(uri):
    return uri.split("/")[-1]

# -----------------------------
# Parse instance types
# -----------------------------
print("Loading instance types...")
g_types = Graph()
g_types.parse(INSTANCE_TYPES_FILE, format="ttl")

nodes = {}

for s, p, o in tqdm(g_types, desc="Parsing types"):
    entity = clean_uri(str(s))
    entity_type = clean_uri(str(o))

    if entity_type in ALLOWED_TYPES:
        nodes[entity] = entity_type

print(f"Collected {len(nodes)} typed entities")

# -----------------------------
# Parse relationships
# -----------------------------
print("Loading relationships...")
g_rel = Graph()
g_rel.parse(RELATIONS_FILE, format="ttl")

relationships = []

for s, p, o in tqdm(g_rel, desc="Parsing relations"):
    subj = clean_uri(str(s))
    pred = clean_uri(str(p))
    obj = clean_uri(str(o))

    if pred in ALLOWED_RELATIONS:
        if subj in nodes and obj in nodes:
            relationships.append({
                "subject": subj,
                "predicate": pred,
                "object": obj
            })

print(f"Collected {len(relationships)} relationships")

# -----------------------------
# Write CSVs
# -----------------------------
nodes_df = pd.DataFrame(
    [{"id": k, "type": v} for k, v in nodes.items()]
)

rels_df = pd.DataFrame(relationships)

nodes_df.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)
rels_df.to_csv(os.path.join(OUT_DIR, "relationships.csv"), index=False)

print("Done. Files written to data/processed/")
