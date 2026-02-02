# src/evaluation/evaluate_nlm_gene.py
"""
Evaluate the RAG pipeline on the NLM Gene entity linking dataset.

The dataset links gene mentions to NCBI Gene IDs (NCBIGene:XXXXX).
The ontology source is gene_info.tsv from NCBI.

Usage:
  python -m src.evaluation.evaluate_nlm_gene --ingest   # Build indexes (first run)
  python -m src.evaluation.evaluate_nlm_gene --limit 2  # Test with 2 entities
  python -m src.evaluation.evaluate_nlm_gene            # Full evaluation
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
# CONFIGURATION - NLM Gene / NCBI Gene
# ============================================================

ONTOLOGY_KEY = "ncbi_gene"

# Subfolder for all generated artifacts
ONTOLOGY_DIR = DATA_DIR / ONTOLOGY_KEY

# Override global config for this ontology
config.ONTOLOGIES_CONFIG = {
    ONTOLOGY_KEY: {
        # Path to gene_info.tsv (NCBI Gene database dump)
        'path': DATA_DIR / "ontologies" / "gene_info.tsv",
        
        # CURIE prefix for gene IDs
        'prefix': 'NCBIGene:',
        
        # Regex pattern for valid IDs
        'id_pattern': r'^NCBIGene:\d+$',
        
        # Generated artifact paths (all in ONTOLOGY_DIR subfolder)
        'dump_json_path': ONTOLOGY_DIR / "ontology_dump.json",
        'enriched_docs_path': ONTOLOGY_DIR / "enriched_documents.json",
        'embeddings_path': ONTOLOGY_DIR / "embeddings.json",
        'whoosh_index_dir': ONTOLOGY_DIR / "whoosh_index",
        'faiss_index_path': ONTOLOGY_DIR / "faiss_index.bin",
        'faiss_metadata_path': ONTOLOGY_DIR / "faiss_metadata.json",
    }
}

# Restrict pipeline to this ontology only
config.RESTRICT_TARGET_ONTOLOGIES = [ONTOLOGY_KEY]

# Add NCBIGene to CURIE prefix map if not present
if "NCBIGene" not in config.CURIE_PREFIX_MAP.values():
    config.CURIE_PREFIX_MAP["https://www.ncbi.nlm.nih.gov/gene/"] = "NCBIGene"

# Override prompts with gene-specific versions
config.SELECTOR_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "gene_selection.tpl"
config.CONFIDENCE_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "gene_confidence.tpl"
config.SYNONYM_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "gene_synonyms.tpl"

# Dataset paths
DATASET_DIR = DATA_DIR / "datasets" / "nlm_gene"
INPUT_FILE = DATASET_DIR / "test.jsonl.gz"
OUTPUT_FILE = ONTOLOGY_DIR / "results.json"

# Cache path
CACHE_PATH = ONTOLOGY_DIR / "cache.json"

# ============================================================
# INGESTION CONFIGURATION
# ============================================================

# Filter to specific organism(s) by tax_id
# Mouse = 10090, Human = 9606, Rat = 10116
# Set to None to include all organisms (WARNING: 65M+ entries!)
FILTER_TAX_IDS = [10090]  # Mouse only - matches NLM Gene test set

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
    """
    Build indexes for NCBI Gene.
    
    Uses the generic TSV parser with gene_info.tsv-specific column mappings.
    """
    from src.ingestion import parse_tsv, build_whoosh_index, build_embeddings, build_faiss_index
    
    cfg = config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]
    
    print(f"\n{'='*60}")
    print(f"INGESTING: NCBI Gene (gene_info.tsv)")
    print(f"{'='*60}")
    
    if FILTER_TAX_IDS:
        print(f"Filtering to tax_ids: {FILTER_TAX_IDS}")
    else:
        print("WARNING: No tax_id filter - this will process 65M+ entries!")
    
    # Define filter function for organism
    def filter_by_taxid(row: Dict[str, str]) -> bool:
        if not FILTER_TAX_IDS:
            return True
        tax_id = row.get("tax_id", "").strip()
        try:
            return int(tax_id) in FILTER_TAX_IDS
        except ValueError:
            return False
    
    print(f"\n[1/4] Parsing gene_info.tsv...")
    print(f"       Source: {cfg['path']}")
    
    if not cfg['path'].exists():
        raise FileNotFoundError(
            f"gene_info.tsv not found at: {cfg['path']}\n"
            f"Download from: https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
        )
    
    parse_tsv(
        tsv_path=cfg['path'],
        output_path=cfg['dump_json_path'],
        id_column="GeneID",
        label_column="Symbol",
        id_prefix="NCBIGene:",
        synonyms_column="Synonyms",
        synonyms_separator="|",
        definition_column="description",
        filter_func=filter_by_taxid,
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
    """
    Load NLM Gene test dataset (JSONL.gz format).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    
    items = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Build context from left/right context if available
            context_left = item.get("context_left", "")
            context_right = item.get("context_right", "")
            mention = item["mention"]
            
            if context_left or context_right:
                context = f"{context_left} [{mention}] {context_right}".strip()
            else:
                context = ""
            
            items.append({
                "id": item["mention_id"],
                "query": mention,
                "context": context,
                "gold_ids": item["gold_ids"],
                "doc_id": item.get("doc_id", ""),
            })
            
            if limit and len(items) >= limit:
                break
    
    logger.info(f"Loaded {len(items)} items from {input_path}")
    return items


# ============================================================
# EVALUATION HELPERS
# ============================================================

def simplify_mapping_result(result: Optional[Dict]) -> Optional[Dict]:
    """Remove verbose keys from a mapping result dictionary."""
    if not isinstance(result, dict):
        return result
    keys_to_remove = {'ancestors', 'parents', 'relations'}
    return {k: v for k, v in result.items() if k not in keys_to_remove}


def normalize_gene_id(pred_id: Optional[str]) -> Optional[str]:
    """Normalize predicted ID to NCBIGene:XXXXX format."""
    if not pred_id:
        return None
    pred_id = pred_id.strip()
    # Already in correct format
    if pred_id.startswith("NCBIGene:"):
        return pred_id
    # Just a number
    if pred_id.isdigit():
        return f"NCBIGene:{pred_id}"
    return pred_id


async def evaluate_item(
    pipeline,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single mention using the pipeline."""
    query = item["query"]
    context = item.get("context", "")
    item_id = item["id"]
    gold_ids = set(item.get("gold_ids", []))
    
    # Check cache
    cache_key = f"{query}|||{context[:100]}" if context else query
    if cache_key in cache:
        logger.debug(f"Cache hit: {query}")
        result = cache[cache_key]
    else:
        result, candidates = await pipeline.run(
            query=query,
            context=context,
            semaphore=semaphore,
            target_ontologies=[ONTOLOGY_KEY],
        )
        # Only cache high-confidence results
        if result and isinstance(result, dict):
            conf = result.get('confidence_score', 0)
            if conf and conf >= config.CONFIDENCE_THRESHOLD:
                cache[cache_key] = (result, candidates)
        result = (result, candidates)
    
    # Unpack result
    if isinstance(result, tuple):
        mapping_result, candidates = result
    else:
        mapping_result, candidates = result, []
    
    # Extract predicted ID
    pred_id = None
    if mapping_result and isinstance(mapping_result, dict):
        pred_id = normalize_gene_id(mapping_result.get("id"))
    
    # Check if correct
    is_correct = pred_id in gold_ids if pred_id else False
    
    simplified = simplify_mapping_result(mapping_result)
    
    return {
        "id": item_id,
        "query": query,
        "context": context[:200] if context else "",
        "gold_ids": list(gold_ids),
        "predicted_id": pred_id,
        "is_correct": is_correct,
        "confidence": mapping_result.get("confidence_score") if mapping_result else None,
        "mapping_result": simplified if simplified else "No mapping found",
        "candidates": [c.get("id") for c in candidates[:5]] if candidates else [],
    }


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on NLM Gene dataset"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline first (build indexes)",
    )
    parser.add_argument(
        "--ingest-limit",
        type=int,
        default=None,
        help="Limit rows during ingestion (for testing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test items to evaluate",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override max concurrent requests",
    )
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable DEBUG logging for deep traceability",
    )
    
    args = parser.parse_args()
    
    # Setup logging with appropriate level
    config.LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    setup_run_logging(f"evaluate_{ONTOLOGY_KEY}")
    logger = logging.getLogger(__name__)
    
    # Handle ingestion mode
    if args.ingest:
        run_ingestion(max_rows=args.ingest_limit)
        return
    
    # --- EVALUATION MODE ---
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: NLM Gene → NCBI Gene")
    print(f"Pipeline: {config.PIPELINE}")
    print(f"{'='*60}\n")
    
    # Load cache
    cache: Dict[str, Any] = {}
    if not args.no_cache and CACHE_PATH.exists():
        cache = load_cache(CACHE_PATH)
        logger.info(f"Loaded {len(cache)} cached results")
    
    pipeline = None
    try:
        # Load dataset
        items = load_dataset(INPUT_FILE, limit=args.limit)
        print(f"Loaded {len(items)} test mentions")
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = create_pipeline(config.PIPELINE)
        
        # Setup concurrency
        max_conc = args.max_concurrency or config.MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(max_conc)
        
        # Process items
        tasks = [
            evaluate_item(pipeline, item, semaphore, cache)
            for item in items
        ]
        
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
        
        # Show sample results
        print(f"\n--- Sample Results ---")
        for r in results[:5]:
            status = "✅" if r["is_correct"] else "❌"
            print(f"{status} '{r['query']}' → {r['predicted_id']} (gold: {r['gold_ids']})")
        
        # Save results
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({
                "metrics": {"total": total, "correct": correct, "accuracy": accuracy},
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n❌ Error: {e}")
        print(f"\nTo get started:")
        print(f"  1. Ensure gene_info.tsv is at: {config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]['path']}")
        print(f"  2. Run: python -m src.evaluation.evaluate_nlm_gene --ingest")
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise
        
    finally:
        if pipeline:
            pipeline.close()
        if not args.no_cache:
            save_cache(CACHE_PATH, cache)
            logger.info(f"Saved {len(cache)} cached results")


if __name__ == "__main__":
    asyncio.run(main())
