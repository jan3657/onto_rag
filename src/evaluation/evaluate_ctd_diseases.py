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
import time
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
# OUTPUT_FILE and CACHE_PATH are now dynamic based on model - see main()

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
    start_time = time.perf_counter()
    try:
        query = item["query"]
        gold_ids = set(item["gold_ids"])
        
        # Check cache
        cache_key = query
        from_cache = False
        
        if cache_key in cache:
            result = cache[cache_key]
            from_cache = True
            retrieval_meta = None
        else:
            result, candidates, retrieval_meta = await pipeline.run(
                query=query,
                context="",
                semaphore=semaphore,
                target_ontologies=[ONTOLOGY_KEY],
                gold_ids=gold_ids,
            )
            # Only cache high-confidence results
            if result and isinstance(result, dict):
                conf = result.get('confidence_score', 0)
                if conf and conf >= config.CONFIDENCE_THRESHOLD:
                    cache[cache_key] = (result, candidates, retrieval_meta)
            result = (result, candidates, retrieval_meta)
            
        # Unpack - handle tuple (from runtime) or list (from JSON cache)
        if isinstance(result, (tuple, list)):
            mapping_result = result[0]
            candidates = result[1] if len(result) > 1 else []
            retrieval_meta = result[2] if len(result) > 2 else None
        else:
            mapping_result = result
            candidates = []
            retrieval_meta = None
        
        # Robustness: if mapping_result is a list (rare LLM output edge case), take first item
        if isinstance(mapping_result, list):
            if mapping_result:
                mapping_result = mapping_result[0]
            else:
                mapping_result = {}

        pred_id = mapping_result.get("id") if isinstance(mapping_result, dict) else None
        
        # Correctness check (exact string match for now)
        is_correct = pred_id in gold_ids if pred_id else False
        elapsed_time = time.perf_counter() - start_time
        candidate_ids = [c.get("id") for c in candidates[:10]] if candidates else []
        gold_in_candidates = bool(gold_ids & set(candidate_ids)) if candidate_ids else False
        
        return {
            "query": query,
            "gold_ids": list(gold_ids),
            "predicted_id": pred_id,
            "is_correct": is_correct,
            "confidence": mapping_result.get("confidence_score") if isinstance(mapping_result, dict) else None,
            "candidate_labels": [c.get("label") for c in candidates[:3]] if candidates else [],
            "gold_in_candidates": gold_in_candidates,
            "gold_first_found_at_attempt": retrieval_meta.get("gold_first_found_at") if retrieval_meta else None,
            "total_retrieval_attempts": retrieval_meta.get("total_attempts", 1) if retrieval_meta else 1,
            "time_seconds": round(elapsed_time, 3),
            "from_cache": from_cache,
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"Error evaluating item '{item.get('query')}': {e}", exc_info=True)
        return {
            "query": item.get("query", "unknown"),
            "is_correct": False,
            "error": str(e),
            "gold_first_found_at_attempt": None,
            "total_retrieval_attempts": 0,
            "time_seconds": round(elapsed_time, 3),
            "from_cache": False,
        }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ingest-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--provider", type=str, default=config.PIPELINE,
                        choices=["gemini", "vllm", "ollama"],
                        help="LLM provider to use (default: from config.PIPELINE)")
    args = parser.parse_args()
    
    # Update config.PIPELINE to match CLI arg (affects logging and model naming)
    config.PIPELINE = args.provider
    
    # Setup logging with appropriate level
    config.LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    setup_run_logging(f"evaluate_{ONTOLOGY_KEY}")
    logger = logging.getLogger(__name__)
    
    if args.ingest:
        run_ingestion(args.ingest_limit)
        return
        
    # Evaluation
    print(f"\n{'='*60}\nEVALUATING: CTD Diseases ({args.provider})\n{'='*60}")
    
    # Build dynamic paths based on MODEL for separate results per model
    from src.utils.model_utils import get_model_file_suffix
    model_suffix = get_model_file_suffix()
    output_file = ONTOLOGY_DIR / f"results_{model_suffix}.json"
    cache_path = ONTOLOGY_DIR / f"cache_{model_suffix}.json"
    
    cache = load_cache(cache_path) if not args.no_cache and cache_path.exists() else {}
    items = load_dataset(INPUT_FILE, limit=args.limit)
    
    pipeline = create_pipeline(args.provider)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    
    try:
        start_eval_time = time.perf_counter()
        tasks = [evaluate_item(pipeline, i, semaphore, cache) for i in items]
        
        # Use asyncio.gather with return_exceptions (tqdm.asyncio.gather doesn't support it)
        print(f"Evaluating {len(tasks)} items...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_clock_time = time.perf_counter() - start_eval_time
        
        # Filter out exceptions and log them
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Task {i} failed with exception: {r}")
            else:
                valid_results.append(r)
        results = valid_results
        
        # Calculate detailed metrics
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Timing metrics
        # 'time_seconds' from evaluating an item includes semaphore wait time + processing time.
        # This is effectively "Latency" from the task's perspective.
        individual_times = [r.get("time_seconds", 0) for r in results if not r.get("from_cache")]
        avg_latency = sum(individual_times) / len(individual_times) if individual_times else 0.0
        
        # Throughput: Items processed per second (wall clock)
        items_per_second = total / wall_clock_time if wall_clock_time > 0 else 0.0

        # Retrieval metrics
        retrieval_success = sum(1 for r in results if r.get("gold_in_candidates"))
        retrieval_rate = retrieval_success / total if total > 0 else 0.0

        # Confidence metrics
        confidences = [r.get("confidence") for r in results if r.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        high_conf_count = sum(1 for c in confidences if c >= config.CONFIDENCE_THRESHOLD)
        high_conf_results = [r for r in results if (r.get("confidence") or 0) >= config.CONFIDENCE_THRESHOLD]
        high_conf_correct = sum(1 for r in high_conf_results if r["is_correct"])
        high_conf_precision = high_conf_correct / len(high_conf_results) if high_conf_results else 0.0

        # Error counts
        error_count = sum(1 for r in results if r.get("error"))
        no_prediction_count = sum(1 for r in results if r.get("predicted_id") is None and not r.get("error"))
        cache_hits = sum(1 for r in results if r.get("from_cache"))

        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_time_seconds": round(wall_clock_time, 2),
            "items_per_second": round(items_per_second, 2),
            "avg_latency_seconds": round(avg_latency, 3),
            "concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
            "retrieval_success_count": retrieval_success,
            "retrieval_success_rate": round(retrieval_rate, 4),
            **{f"retrieval_success_rate@{k}": round(
                sum(1 for r in results 
                    if r.get("gold_first_found_at_attempt") is not None 
                    and r.get("gold_first_found_at_attempt") <= k) / total, 4
            ) if total > 0 else 0.0 for k in range(1, config.MAX_PIPELINE_LOOPS + 1)},
            "avg_confidence": round(avg_confidence, 4),
            "high_confidence_count": high_conf_count,
            "high_confidence_precision": round(high_conf_precision, 4),
            "error_count": error_count,
            "no_prediction_count": no_prediction_count,
            "cache_hits": cache_hits
        }

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
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
            
        print(f"\nResults saved to: {output_file}")
            
    finally:
        if 'pipeline' in locals() and pipeline:
            pipeline.close()
        if not args.no_cache:
            save_cache(cache_path, cache)
        
        logging.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
