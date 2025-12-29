import os
import pickle
from pathlib import Path
from bioel.ontology import BiomedicalOntology
from bioel.dataset import Dataset

# Cache directory for processed data
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

print("--- 1. Loading MEDIC Ontology ---")
# This loads the 'MEDIC' ontology (Merged Disease Vocabulary)
# BioEL should automatically download the necessary files if they are missing.
medic_ontology = BiomedicalOntology.load_medic("./data/CTD_diseases.tsv") 
print(f"Loaded MEDIC Ontology with {len(medic_ontology.entities)} entities.")

print("\n--- 2. Loading NCBI-Disease Dataset ---")
dataset_name = "ncbi_disease"
cache_file = CACHE_DIR / f"{dataset_name}_dataset.pkl"

if cache_file.exists():
    print(f"Loading from cache: {cache_file}")
    with open(cache_file, "rb") as f:
        dataset = pickle.load(f)
else:
    print("Downloading and processing dataset (first run)...")
    dataset = Dataset(dataset_name=dataset_name)
    # Save to cache for next time
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved to cache: {cache_file}")
print("Dataset loaded successfully!")

print("\n--- 3. Inspecting the Data ---")
# Inspect the training split - data is stored in dataset.df with a 'split' column
train_data = dataset.df[dataset.df["split"] == "train"]
print(f"Training mentions: {len(train_data)}")

# Show a sample mention from the training data
first_mention = train_data.iloc[0]
print(f"Sample Text: {first_mention['text']}")
print(f"DB IDs: {first_mention['db_ids']}")