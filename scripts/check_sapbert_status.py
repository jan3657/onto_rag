#!/usr/bin/env python3
"""
Diagnostic script to check SapBERT integration status across all datasets.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ontologies"

DATASETS = {
    "chebi": DATA_DIR / "chebi",
    "ctd_diseases": DATA_DIR / "ctd_diseases",
    "ncbi_gene": DATA_DIR / "nlm_gene",  # different dir name
    "foodon": DATA_DIR / "cafeteria_foodon",  # different dir name
}

print("=" * 80)
print("SAPBERT INTEGRATION DIAGNOSTIC")
print("=" * 80)

for ds_name, ds_path in DATASETS.items():
    print(f"\n📁 {ds_name.upper()} ({ds_path})")
    print("-" * 60)

    if not ds_path.exists():
        print(f"   ❌ Directory not found!")
        continue

    # Check for required files
    files_to_check = {
        "MiniLM embeddings": ds_path / "embeddings_minilm.json",
        "MiniLM FAISS": ds_path / "faiss_index_minilm.bin",
        "SapBERT embeddings": ds_path / "embeddings_sapbert.json",
        "SapBERT FAISS": ds_path / "faiss_index_sapbert.bin",
        "Whoosh index": ds_path / "whoosh_index",
        "Legacy embeddings": ds_path / "embeddings.json",  # backward compat
    }

    status = {}
    for name, path in files_to_check.items():
        exists = path.exists()
        status[name] = exists
        icon = "✅" if exists else "❌"

        size_info = ""
        if exists:
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                size_info = f" ({size_mb:.1f} MB)"
            elif path.is_dir():
                # Count files in whoosh index
                file_count = len(list(path.glob("**/*")))
                size_info = f" ({file_count} files)"

        print(f"   {icon} {name:25s} {size_info}")

    # Summary
    has_minilm = status.get("MiniLM embeddings") and status.get("MiniLM FAISS")
    has_sapbert = status.get("SapBERT embeddings") and status.get("SapBERT FAISS")
    has_lexical = status.get("Whoosh index")

    print(f"\n   Summary:")
    print(f"   - Lexical:  {'✅ Ready' if has_lexical else '❌ Missing'}")
    print(f"   - MiniLM:   {'✅ Ready' if has_minilm else '❌ Missing'}")
    print(f"   - SapBERT:  {'✅ Ready' if has_sapbert else '⚠️  MISSING - Need to run ingestion!'}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# Check if ANY dataset has SapBERT
any_sapbert = False
for ds_name, ds_path in DATASETS.items():
    if (ds_path / "embeddings_sapbert.json").exists():
        any_sapbert = True
        break

if not any_sapbert:
    print("""
⚠️  CRITICAL ISSUE: No SapBERT embeddings found for ANY dataset!

ROOT CAUSE:
===========
The evaluation scripts were updated to generate SapBERT embeddings,
but ingestion was NOT re-run after the update.

SOLUTION:
=========
Re-run ingestion for each dataset using the updated scripts:

1. ChEBI (chemicals):
   python -m src.evaluation.evaluate_craft_chebi --ingest

2. NLM Gene (genes):
   python -m src.evaluation.evaluate_nlm_gene --ingest

3. CTD Diseases (diseases):
   python -m src.evaluation.evaluate_ctd_diseases --ingest

4. FoodOn (food):
   python -m src.evaluation.evaluate_cafeteria_foodon --ingest

⏱️  NOTE: Each ingestion will take ~2-3x longer than before since it
   generates TWO sets of embeddings (MiniLM + SapBERT).

📊 PRIORITY ORDER (by expected SapBERT benefit):
   1. NLM Gene (specialized biomedical terminology)
   2. CTD Diseases (medical terminology)
   3. ChEBI (already done for testing)
   4. FoodOn (less specialized, lower priority)
""")
else:
    print("✅ Some datasets have SapBERT embeddings!")
    print("\nDatasets ready for 3-way retrieval:")
    for ds_name, ds_path in DATASETS.items():
        if (ds_path / "embeddings_sapbert.json").exists():
            print(f"  ✅ {ds_name}")
    print("\nDatasets missing SapBERT (need re-ingestion):")
    for ds_name, ds_path in DATASETS.items():
        if not (ds_path / "embeddings_sapbert.json").exists():
            print(f"  ❌ {ds_name}")
