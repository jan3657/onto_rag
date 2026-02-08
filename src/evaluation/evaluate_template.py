# src/evaluation/evaluate_template.py
"""
Template evaluation script for new datasets/ontologies.

Copy this file and customize the CONFIGURATION section for your use case.

Usage:
  python -m src.evaluation.evaluate_template --ingest   # First run: build indexes
  python -m src.evaluation.evaluate_template            # Subsequent runs: evaluate
  python -m src.evaluation.evaluate_template --limit 10 # Evaluate subset
"""

import argparse
import asyncio
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
ONTOLOGIES_DIR = PROJECT_ROOT / "ontologies"

# ============================================================
# CONFIGURATION - Customize for your dataset/ontology
# ============================================================

# Unique key for this ontology (used in file naming)
ONTOLOGY_KEY = "my_ontology"

# Override global config for this ontology
config.ONTOLOGIES_CONFIG = {
    ONTOLOGY_KEY: {
        # Path to OWL/OBO ontology file
        'path': ONTOLOGIES_DIR / f"{ONTOLOGY_KEY}.owl",

        # CURIE prefix for this ontology (used for ID matching)
        'prefix': 'MYONT:',

        # Regex pattern for valid IDs
        'id_pattern': r'^MYONT:\d+$',

        # Generated artifact paths (created by --ingest)
        'dump_json_path': DATA_DIR / f"ontology_dump_{ONTOLOGY_KEY}.json",
        'enriched_docs_path': DATA_DIR / f"enriched_documents_{ONTOLOGY_KEY}.json",
        'embeddings_path': DATA_DIR / f"embeddings_{ONTOLOGY_KEY}.json",
        'whoosh_index_dir': DATA_DIR / f"whoosh_index_{ONTOLOGY_KEY}",
        'faiss_index_path': DATA_DIR / f"faiss_index_{ONTOLOGY_KEY}.bin",
        'faiss_metadata_path': DATA_DIR / f"faiss_metadata_{ONTOLOGY_KEY}.json",
    }
}

# Restrict pipeline to this ontology only
config.RESTRICT_TARGET_ONTOLOGIES = [ONTOLOGY_KEY]

# Optional: Override prompts for this domain
# Uncomment and customize if you have domain-specific prompts
# config.SELECTOR_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / f"{ONTOLOGY_KEY}_selection.tpl"
# config.CONFIDENCE_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / f"{ONTOLOGY_KEY}_confidence.tpl"
# config.SYNONYM_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / f"{ONTOLOGY_KEY}_synonyms.tpl"

# Optional: Override embedding model for this ontology
# config.EMBEDDING_MODEL_NAME = "custom-model"

# Dataset-specific paths
INPUT_FILE = DATA_DIR / f"{ONTOLOGY_KEY}_dataset.json"
# OUTPUT_FILE and CACHE_PATH are now dynamic based on model - see main()

# ============================================================
# LOGGING SETUP
# ============================================================

# Default log level (can be overridden by --debug CLI flag)
# This will be set after argument parsing
logger = logging.getLogger(__name__)


# ============================================================
# INGESTION (run with --ingest flag)
# ============================================================

def run_ingestion() -> None:
    """Build all indexes for this ontology. Run once before evaluation."""
    from src.ingestion import parse_ontology, build_whoosh_index, build_embeddings, build_faiss_index

    cfg = config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]

    print(f"\n{'='*60}")
    print(f"INGESTING ONTOLOGY: {ONTOLOGY_KEY}")
    print(f"{'='*60}\n")

    print(f"[1/4] Parsing ontology: {cfg['path']}")
    if not cfg['path'].exists():
        raise FileNotFoundError(f"Ontology file not found: {cfg['path']}")
    parse_ontology(cfg['path'], cfg['dump_json_path'])

    print(f"\n[2/4] Building Whoosh index...")
    build_whoosh_index(cfg['dump_json_path'], cfg['whoosh_index_dir'])

    print(f"\n[3/4] Building embeddings...")
    build_embeddings(cfg['dump_json_path'], cfg['embeddings_path'])

    print(f"\n[4/4] Building FAISS index...")
    build_faiss_index(cfg['embeddings_path'], cfg['faiss_index_path'], cfg['faiss_metadata_path'])

    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nGenerated artifacts:")
    print(f"  - Ontology dump: {cfg['dump_json_path']}")
    print(f"  - Whoosh index:  {cfg['whoosh_index_dir']}")
    print(f"  - Embeddings:    {cfg['embeddings_path']}")
    print(f"  - FAISS index:   {cfg['faiss_index_path']}")
    print(f"\nYou can now run evaluation without --ingest flag.\n")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def simplify_mapping_result(result: Optional[Dict]) -> Optional[Dict]:
    """Remove verbose keys from a mapping result dictionary."""
    if not isinstance(result, dict):
        return result
    keys_to_remove = {'ancestors', 'parents', 'relations'}
    return {k: v for k, v in result.items() if k not in keys_to_remove}


def load_dataset(input_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.

    Expected format (customize as needed):
    [
        {"id": "item1", "query": "search term", "context": "optional context"},
        ...
    ]

    Or dict format:
    {
        "item1": {"query": "search term", ...},
        ...
    }
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to list format
    if isinstance(data, dict):
        items = [{"id": k, **v} for k, v in data.items()]
    else:
        items = data

    if limit:
        items = items[:limit]

    logger.info(f"Loaded {len(items)} items from {input_path}")
    return items


# ============================================================
# MAIN EVALUATION
# ============================================================

async def evaluate_item(
    pipeline,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single item using the pipeline."""
    query = item.get("query", item.get("text", ""))
    context = item.get("context", "")
    item_id = item.get("id", query)

    # Check cache
    cache_key = f"{query}|||{context[:100]}" if context else query
    if cache_key in cache:
        logger.debug(f"Cache hit: {query}")
        result = cache[cache_key]
    else:
        result, candidates = await pipeline.run(
            query=query,
            context=context,
            lexical_k=config.DEFAULT_K_LEXICAL,
            minilm_k=config.DEFAULT_K_MINILM,
            sapbert_k=config.DEFAULT_K_SAPBERT,
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

    simplified = simplify_mapping_result(mapping_result)

    return {
        "id": item_id,
        "query": query,
        "context": context,
        "mapping_result": simplified if simplified else "No mapping found",
        "candidates": candidates[:5] if candidates else [],  # Limit candidates in output
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate RAG pipeline on {ONTOLOGY_KEY} dataset"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline first (build indexes)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (don't read or write)",
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
    parser.add_argument(
        "--provider",
        type=str,
        default=config.PIPELINE,
        choices=["gemini", "vllm", "ollama"],
        help="LLM provider to use (default: from config.PIPELINE)",
    )

    args = parser.parse_args()

    # Update config.PIPELINE to match CLI arg (affects logging and model naming)
    config.PIPELINE = args.provider

    # Setup logging with appropriate level
    config.LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    setup_run_logging(f"evaluate_{ONTOLOGY_KEY}")
    logger = logging.getLogger(__name__)

    # Handle ingestion mode
    if args.ingest:
        run_ingestion()
        return

    # --- EVALUATION MODE ---

    print(f"\n{'='*60}")
    print(f"EVALUATING: {ONTOLOGY_KEY} ({args.provider})")
    print(f"{'='*60}\n")

    # Build dynamic paths based on MODEL for separate results per model
    from src.utils.model_utils import get_model_file_suffix
    model_suffix = get_model_file_suffix()
    output_file = DATA_DIR / "outputs" / f"results_{ONTOLOGY_KEY}_{model_suffix}.json"
    cache_path = DATA_DIR / f"cache_{ONTOLOGY_KEY}_{model_suffix}.json"

    # Load cache
    cache: Dict[str, Any] = {}
    if not args.no_cache and cache_path.exists():
        cache = load_cache(cache_path)
        logger.info(f"Loaded {len(cache)} cached results")

    pipeline = None
    try:
        # Load dataset
        items = load_dataset(INPUT_FILE, limit=args.limit)

        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = create_pipeline(args.provider)

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

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Evaluation complete!")
        print(f"   Processed: {len(results)} items")
        print(f"   Results saved to: {output_file}")

    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure to:")
        print(f"  1. Place your ontology at: {config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]['path']}")
        print(f"  2. Run ingestion: python -m src.evaluation.evaluate_template --ingest")
        print(f"  3. Place your dataset at: {INPUT_FILE}")

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise

    finally:
        if pipeline:
            pipeline.close()
        if not args.no_cache:
            save_cache(cache_path, cache)
            logger.info(f"Saved {len(cache)} cached results")


if __name__ == "__main__":
    asyncio.run(main())
