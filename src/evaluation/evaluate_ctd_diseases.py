# src/evaluation/evaluate_ctd_diseases.py
"""
Evaluate the RAG pipeline on the NCBI Disease dataset.

The dataset links disease mentions to MeSH/OMIM IDs via the CTD Diseases ontology.
The ontology source is CTD_diseases.tsv.

Usage:
  python -m src.evaluation.evaluate_ctd_diseases --ingest   # Build indexes
  python -m src.evaluation.evaluate_ctd_diseases --limit 5  # Test
  python -m src.evaluation.evaluate_ctd_diseases            # Full evaluation
"""

import argparse
import asyncio
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as asyncio_tqdm

from src import config
from src.pipeline import create_pipeline
from src.utils.logging_config import setup_run_logging
from src.utils.cache import load_cache, save_cache

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# CONFIGURATION - CTD Diseases
# ============================================================

ONTOLOGY_KEY = "ctd_diseases"

# Subfolder for artifacts
ONTOLOGY_DIR = DATA_DIR / ONTOLOGY_KEY

# Override global config for this ontology
config.ONTOLOGIES_CONFIG = {
    ONTOLOGY_KEY: {
        'path': DATA_DIR / "ontologies" / "CTD_diseases.tsv",
        # Prefix handling is tricky as file has "MESH:..." but keys need consistency
        # We'll rely on the full CURIEs in the file
        'prefix': 'MESH:', # Primary prefix
        'id_pattern': r'^(MESH|OMIM):.+',
        
        # Generated artifact paths
        'dump_json_path': ONTOLOGY_DIR / "ontology_dump.json",
        'enriched_docs_path': ONTOLOGY_DIR / "enriched_documents.json",
        'embeddings_path': ONTOLOGY_DIR / "embeddings.json",
        'whoosh_index_dir': ONTOLOGY_DIR / "whoosh_index",
        'faiss_index_path': ONTOLOGY_DIR / "faiss_index.bin",
        'faiss_metadata_path': ONTOLOGY_DIR / "faiss_metadata.json",
    }
}

# Restrict pipeline
config.RESTRICT_TARGET_ONTOLOGIES = [ONTOLOGY_KEY]

# Override prompts
config.SELECTOR_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "disease_selection.tpl"
config.CONFIDENCE_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "disease_confidence.tpl"
config.SYNONYM_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "disease_synonyms.tpl"

# Dataset paths
DATASET_DIR = DATA_DIR / "datasets" / "ncbi_disease"
INPUT_FILE = DATASET_DIR / "test.jsonl.gz"
OUTPUT_FILE = ONTOLOGY_DIR / "results.json"
CACHE_PATH = ONTOLOGY_DIR / "cache.json"

# ============================================================
# LOGGING SETUP
# ============================================================

# Default log level (can be overridden by --debug CLI flag)
# This will be set after argument parsing
logger = logging.getLogger(__name__)


# ============================================================
# INGESTION
# ============================================================

def run_ingestion(max_rows: Optional[int] = None) -> None:
    """Build indexes for CTD Diseases."""
    from src.ingestion import parse_tsv, build_whoosh_index, build_embeddings, build_faiss_index
    
    cfg = config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]
    
    print(f"\n{'='*60}")
    print(f"INGESTING: CTD Diseases")
    print(f"{'='*60}")
    
    print(f"\n[1/4] Parsing CTD_diseases.tsv...")
    if not cfg['path'].exists():
        raise FileNotFoundError(f"Ontology file not found: {cfg['path']}")
    
    # Custom transform to handle synonyms and prefixes
    def transform_ctd_row(row: Dict[str, str]) -> Dict[str, Any]:
        # Synonyms are pipe-separated
        syn_str = row.get("Synonyms", "")
        synonyms = [s.strip() for s in syn_str.split("|") if s.strip()]
        
        # Alt IDs are useful as synonyms too? Maybe not for lexical matching but useful context
        
        return {
            "label": row.get("DiseaseName", ""),
            "synonyms": synonyms,
            "definition": row.get("Definition", ""),
            "parents": row.get("ParentIDs", "").split("|"),
            "relations": [], # Could extract tree numbers etc if needed
            "relations_text": "",
        }

    parse_tsv(
        tsv_path=cfg['path'],
        output_path=cfg['dump_json_path'],
        id_column="DiseaseID",
        label_column="DiseaseName",
        id_prefix="", # IDs already have prefix (e.g. 'MESH:D000000')
        header_starts_with="# DiseaseName", # Critical for skipping comments
        transform_func=transform_ctd_row,
        max_rows=max_rows,
    )
    
    print(f"\n[2/4] Building Whoosh index...")
    build_whoosh_index(cfg['dump_json_path'], cfg['whoosh_index_dir'])
    
    print(f"\n[3/4] Building embeddings...")
    build_embeddings(cfg['dump_json_path'], cfg['embeddings_path'])
    
    print(f"\n[4/4] Building FAISS index...")
    build_faiss_index(cfg['embeddings_path'], cfg['faiss_index_path'], cfg['faiss_metadata_path'])
    
    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE!")
    print(f"{'='*60}")


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(input_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load NCBI Disease test set."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    
    items = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            items.append({
                "id": str(len(items)), # No native ID in basic jsonl
                "query": item["mention"],
                "context": "", # Not provided/used in this simple format?
                "gold_ids": item["gold_ids"],
            })
            
            if limit and len(items) >= limit:
                break
    
    logger.info(f"Loaded {len(items)} items from {input_path}")
    return items


# ============================================================
# MAIN
# ============================================================

async def evaluate_item(pipeline, item, semaphore, cache):
    """Evaluate single item."""
    try:
        query = item["query"]
        gold_ids = set(item["gold_ids"])
        
        # Check cache
        if query in cache:
            result = cache[query]
        else:
            result, candidates = await pipeline.run(
                query=query,
                context="",
                semaphore=semaphore,
                target_ontologies=[ONTOLOGY_KEY],
            )
            # Only cache high-confidence results
            if result and isinstance(result, dict):
                conf = result.get('confidence_score', 0)
                if conf and conf >= config.CONFIDENCE_THRESHOLD:
                    cache[query] = (result, candidates)
            result = (result, candidates)
            
        # Unpack - handle tuple (from runtime) or list (from JSON cache)
        if isinstance(result, (tuple, list)):
            mapping_result = result[0]
            candidates = result[1] if len(result) > 1 else []
        else:
            mapping_result = result
            candidates = []
        
        # Robustness: if mapping_result is a list (rare LLM output edge case), take first item
        if isinstance(mapping_result, list):
            if mapping_result:
                mapping_result = mapping_result[0]
            else:
                mapping_result = {}

        pred_id = mapping_result.get("id") if isinstance(mapping_result, dict) else None
        
        # Correctness check (exact string match for now)
        is_correct = pred_id in gold_ids if pred_id else False
        
        return {
            "query": query,
            "gold_ids": list(gold_ids),
            "predicted_id": pred_id,
            "is_correct": is_correct,
            "confidence": mapping_result.get("confidence_score") if isinstance(mapping_result, dict) else None,
            "candidate_labels": [c.get("label") for c in candidates[:3]] if candidates else []
        }
    except Exception as e:
        logger.error(f"Error evaluating item '{item.get('query')}': {e}", exc_info=True)
        return {
            "query": item.get("query", "unknown"),
            "is_correct": False,
            "error": str(e)
        }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ingest-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging for deep traceability")
    args = parser.parse_args()
    
    # Setup logging with appropriate level
    config.LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    setup_run_logging(f"evaluate_{ONTOLOGY_KEY}")
    logger = logging.getLogger(__name__)
    
    if args.ingest:
        run_ingestion(args.ingest_limit)
        return
        
    # Evaluation
    print(f"\n{'='*60}\nEVALUATING: CTD Diseases\n{'='*60}")
    
    cache = load_cache(CACHE_PATH) if not args.no_cache and CACHE_PATH.exists() else {}
    items = load_dataset(INPUT_FILE, limit=args.limit)
    
    pipeline = create_pipeline(config.PIPELINE)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    
    try:
        tasks = [evaluate_item(pipeline, i, semaphore, cache) for i in items]
        
        # Use asyncio.gather with return_exceptions (tqdm.asyncio.gather doesn't support it)
        print(f"Evaluating {len(tasks)} items...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Task {i} failed with exception: {r}")
            else:
                valid_results.append(r)
        results = valid_results
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total:    {total}")
        print(f"Correct:  {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Sample
        print(f"\n--- Sample Results ---")
        for r in results[:5]:
             print(f"{'✅' if r['is_correct'] else '❌'} '{r['query']}' → {r['predicted_id']} (Gold: {r['gold_ids']})")
             
        # Save
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w") as f:
            json.dump({
                "metrics": {"total": total, "correct": correct, "accuracy": accuracy},
                "results": results
            }, f, indent=2)
            
        print(f"\nResults saved to: {OUTPUT_FILE}")
            
    finally:
        if 'pipeline' in locals() and pipeline:
            pipeline.close()
        if not args.no_cache:
            save_cache(CACHE_PATH, cache)
        
        logging.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
