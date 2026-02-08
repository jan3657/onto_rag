# src/evaluation/run_benchmark.py
"""
Run benchmark evaluation across multiple datasets.

Evaluates 100 random samples from each dataset (CRAFT ChEBI, CTD Diseases, NLM Gene)
using a fixed random seed for reproducibility across model comparisons.

Usage:
  python -m src.evaluation.run_benchmark                    # Full 100-sample benchmark
  python -m src.evaluation.run_benchmark --limit 5          # Quick test
  python -m src.evaluation.run_benchmark --provider vllm    # Use vLLM instead of Gemini
"""

import argparse
import asyncio
import gzip
import glob
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config
from src.pipeline import create_pipeline
from src.utils.logging_config import setup_run_logging
from src.utils.model_utils import get_model_file_suffix
from src.utils.cache import load_cache, save_cache
from src.evaluation.source_metrics import calculate_source_metrics

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "benchmark_results"

# Default random seed for reproducible sampling
DEFAULT_RANDOM_SEED = 42


# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASETS = {
    "craft_chebi": {
        "name": "CRAFT ChEBI",
        "ontology_key": "craft_chebi",
        "input_file": DATA_DIR / "datasets" / "craft_chebi" / "test.jsonl.gz",
        "ontology_dir": DATA_DIR / "craft_chebi",
        "ontology_config": {
            'path': DATA_DIR / "ontologies" / "chebi.owl",
            'prefix': 'CHEBI:',
            'id_pattern': r'^(CHEBI):\d+',
            'dump_json_path': DATA_DIR / "craft_chebi" / "ontology_dump.json",
            'enriched_docs_path': DATA_DIR / "craft_chebi" / "enriched_documents.json",
            'embeddings_path': DATA_DIR / "craft_chebi" / "embeddings.json",
            'embeddings_minilm_path': DATA_DIR / "craft_chebi" / "embeddings_minilm.json",
            'embeddings_sapbert_path': DATA_DIR / "craft_chebi" / "embeddings_sapbert.json",
            'whoosh_index_dir': DATA_DIR / "craft_chebi" / "whoosh_index",
            'faiss_index_path': DATA_DIR / "craft_chebi" / "faiss_index.bin",
            'faiss_metadata_path': DATA_DIR / "craft_chebi" / "faiss_metadata.json",
            'faiss_index_minilm_path': DATA_DIR / "craft_chebi" / "faiss_index_minilm.bin",
            'faiss_metadata_minilm_path': DATA_DIR / "craft_chebi" / "faiss_metadata_minilm.json",
            'faiss_index_sapbert_path': DATA_DIR / "craft_chebi" / "faiss_index_sapbert.bin",
            'faiss_metadata_sapbert_path': DATA_DIR / "craft_chebi" / "faiss_metadata_sapbert.json",
        },
        "prompts": {
            "selector": PROJECT_ROOT / "prompts" / "food_selection.tpl",
            "confidence": PROJECT_ROOT / "prompts" / "food_confidence.tpl",
            "synonym": PROJECT_ROOT / "prompts" / "food_synonyms.tpl",
        },
        "load_item": lambda item, idx: {
            "id": str(idx),
            "query": item["mention"],
            "context": "",
            "gold_ids": item["gold_ids"],
        },
    },
    "ctd_diseases": {
        "name": "CTD Diseases",
        "ontology_key": "ctd_diseases",
        "input_file": DATA_DIR / "datasets" / "ncbi_disease" / "test.jsonl.gz",
        "ontology_dir": DATA_DIR / "ctd_diseases",
        "ontology_config": {
            'path': DATA_DIR / "ontologies" / "CTD_diseases.tsv",
            'prefix': 'MESH:',
            'id_pattern': r'^(MESH|OMIM):.+',
            'dump_json_path': DATA_DIR / "ctd_diseases" / "ontology_dump.json",
            'enriched_docs_path': DATA_DIR / "ctd_diseases" / "enriched_documents.json",
            'embeddings_path': DATA_DIR / "ctd_diseases" / "embeddings.json",
            'embeddings_minilm_path': DATA_DIR / "ctd_diseases" / "embeddings_minilm.json",
            'embeddings_sapbert_path': DATA_DIR / "ctd_diseases" / "embeddings_sapbert.json",
            'whoosh_index_dir': DATA_DIR / "ctd_diseases" / "whoosh_index",
            'faiss_index_path': DATA_DIR / "ctd_diseases" / "faiss_index.bin",
            'faiss_metadata_path': DATA_DIR / "ctd_diseases" / "faiss_metadata.json",
            'faiss_index_minilm_path': DATA_DIR / "ctd_diseases" / "faiss_index_minilm.bin",
            'faiss_metadata_minilm_path': DATA_DIR / "ctd_diseases" / "faiss_metadata_minilm.json",
            'faiss_index_sapbert_path': DATA_DIR / "ctd_diseases" / "faiss_index_sapbert.bin",
            'faiss_metadata_sapbert_path': DATA_DIR / "ctd_diseases" / "faiss_metadata_sapbert.json",
        },
        "prompts": {
            "selector": PROJECT_ROOT / "prompts" / "disease_selection.tpl",
            "confidence": PROJECT_ROOT / "prompts" / "disease_confidence.tpl",
            "synonym": PROJECT_ROOT / "prompts" / "disease_synonyms.tpl",
        },
        "load_item": lambda item, idx: {
            "id": str(idx),
            "query": item["mention"],
            "context": "",
            "gold_ids": item["gold_ids"],
        },
    },
    "ncbi_gene": {
        "name": "NLM Gene",
        "ontology_key": "ncbi_gene",
        "input_file": DATA_DIR / "datasets" / "nlm_gene" / "test.jsonl.gz",
        "ontology_dir": DATA_DIR / "ncbi_gene",
        "ontology_config": {
            'path': DATA_DIR / "ontologies" / "gene_info.tsv",
            'prefix': 'NCBIGene:',
            'id_pattern': r'^NCBIGene:\d+$',
            'dump_json_path': DATA_DIR / "ncbi_gene" / "ontology_dump.json",
            'enriched_docs_path': DATA_DIR / "ncbi_gene" / "enriched_documents.json",
            'embeddings_path': DATA_DIR / "ncbi_gene" / "embeddings.json",
            'embeddings_minilm_path': DATA_DIR / "ncbi_gene" / "embeddings_minilm.json",
            'embeddings_sapbert_path': DATA_DIR / "ncbi_gene" / "embeddings_sapbert.json",
            'whoosh_index_dir': DATA_DIR / "ncbi_gene" / "whoosh_index",
            'faiss_index_path': DATA_DIR / "ncbi_gene" / "faiss_index.bin",
            'faiss_metadata_path': DATA_DIR / "ncbi_gene" / "faiss_metadata.json",
            'faiss_index_minilm_path': DATA_DIR / "ncbi_gene" / "faiss_index_minilm.bin",
            'faiss_metadata_minilm_path': DATA_DIR / "ncbi_gene" / "faiss_metadata_minilm.json",
            'faiss_index_sapbert_path': DATA_DIR / "ncbi_gene" / "faiss_index_sapbert.bin",
            'faiss_metadata_sapbert_path': DATA_DIR / "ncbi_gene" / "faiss_metadata_sapbert.json",
        },
        "prompts": {
            "selector": PROJECT_ROOT / "prompts" / "gene_selection.tpl",
            "confidence": PROJECT_ROOT / "prompts" / "gene_confidence.tpl",
            "synonym": PROJECT_ROOT / "prompts" / "gene_synonyms.tpl",
        },
        "load_item": lambda item, idx: {
            "id": item.get("mention_id", str(idx)),
            "query": item["mention"],
            "context": f"{item.get('context_left', '')} [{item['mention']}] {item.get('context_right', '')}".strip() if item.get('context_left') or item.get('context_right') else "",
            "gold_ids": item["gold_ids"],
        },
        "normalize_id": lambda pred_id: f"NCBIGene:{pred_id}" if pred_id and pred_id.isdigit() else pred_id,
    },
    "cafeteria_foodon": {
        "name": "Cafeteria FCD (FoodOn)",
        "ontology_key": "cafeteria_foodon",
        "input_file": DATA_DIR / "datasets" / "cafeteria_fcd" / "test.jsonl.gz",
        "ontology_dir": DATA_DIR / "cafeteria_foodon",
        "ontology_config": {
            'path': DATA_DIR / "ontologies" / "foodon.owl",
            'prefix': 'FOODON:',
            'id_pattern': r'^(FOODON):\d+',
            'dump_json_path': DATA_DIR / "cafeteria_foodon" / "ontology_dump.json",
            'enriched_docs_path': DATA_DIR / "cafeteria_foodon" / "enriched_documents.json",
            'embeddings_path': DATA_DIR / "cafeteria_foodon" / "embeddings.json",
            'embeddings_minilm_path': DATA_DIR / "cafeteria_foodon" / "embeddings_minilm.json",
            'embeddings_sapbert_path': DATA_DIR / "cafeteria_foodon" / "embeddings_sapbert.json",
            'whoosh_index_dir': DATA_DIR / "cafeteria_foodon" / "whoosh_index",
            'faiss_index_path': DATA_DIR / "cafeteria_foodon" / "faiss_index.bin",
            'faiss_metadata_path': DATA_DIR / "cafeteria_foodon" / "faiss_metadata.json",
            'faiss_index_minilm_path': DATA_DIR / "cafeteria_foodon" / "faiss_index_minilm.bin",
            'faiss_metadata_minilm_path': DATA_DIR / "cafeteria_foodon" / "faiss_metadata_minilm.json",
            'faiss_index_sapbert_path': DATA_DIR / "cafeteria_foodon" / "faiss_index_sapbert.bin",
            'faiss_metadata_sapbert_path': DATA_DIR / "cafeteria_foodon" / "faiss_metadata_sapbert.json",
        },
        "prompts": {
            "selector": PROJECT_ROOT / "prompts" / "food_selection.tpl",
            "confidence": PROJECT_ROOT / "prompts" / "food_confidence.tpl",
            "synonym": PROJECT_ROOT / "prompts" / "food_synonyms.tpl",
        },
        "load_item": lambda item, idx: {
            "id": item.get("mention_id", str(idx)),
            "query": item["mention"],
            "context": f"{item.get('context_left', '')} [{item['mention']}] {item.get('context_right', '')}".strip()
                       if item.get('context_left') or item.get('context_right') else "",
            "gold_ids": item["gold_ids"],
        },
    },
}


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset_raw(input_path: Path) -> List[Dict[str, Any]]:
    """Load raw JSONL.gz dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    items = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def find_latest_run_dir(model_suffix: str) -> Optional[Path]:
    """Find the most recent run directory for the given model."""
    if not RESULTS_DIR.exists():
        return None

    # Look for directories matching pattern: YYYYMMDD_HHMMSS_{model_suffix}
    # We just look for *_{model_suffix} and sort
    candidates = list(RESULTS_DIR.glob(f"*_{model_suffix}"))
    if not candidates:
        return None

    # Sort by name (which starts with timestamp) descending
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]



def sample_items(items: List[Dict], sample_size: Optional[int], seed: int) -> List[Dict]:
    """Randomly sample items with fixed seed. If sample_size is None, return all items."""
    if sample_size is None:
        return items
    random.seed(seed)
    if len(items) <= sample_size:
        return items
    return random.sample(items, sample_size)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

def configure_for_dataset(dataset_key: str) -> None:
    """Configure global config for a specific dataset."""
    ds = DATASETS[dataset_key]

    # Set ontology config
    config.ONTOLOGIES_CONFIG = {
        ds["ontology_key"]: ds["ontology_config"]
    }
    config.RESTRICT_TARGET_ONTOLOGIES = [ds["ontology_key"]]

    # Set prompts
    config.SELECTOR_PROMPT_TEMPLATE_PATH = ds["prompts"]["selector"]
    config.CONFIDENCE_PROMPT_TEMPLATE_PATH = ds["prompts"]["confidence"]
    config.SYNONYM_PROMPT_TEMPLATE_PATH = ds["prompts"]["synonym"]


# ============================================================
# EVALUATION
# ============================================================

async def evaluate_item(
    pipeline,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    cache: Dict[str, Any],
    ontology_key: str,
    normalize_id=None,
) -> Dict[str, Any]:
    """Evaluate single item with comprehensive metrics for paper-quality analysis."""
    import time
    start_time = time.perf_counter()

    try:
        query = item["query"]
        context = item.get("context", "")
        gold_ids = set(item["gold_ids"])

        # Cache key
        cache_key = f"{query}|||{context[:100]}" if context else query
        from_cache = False

        if cache_key in cache:
            result = cache[cache_key]
            from_cache = True
            retrieval_meta = None
        else:
            result, candidates, retrieval_meta = await pipeline.run(
                query=query,
                context=context,
                lexical_k=config.DEFAULT_K_LEXICAL,
                minilm_k=config.DEFAULT_K_MINILM,
                sapbert_k=config.DEFAULT_K_SAPBERT,
                semaphore=semaphore,
                target_ontologies=[ontology_key],
                gold_ids=gold_ids,
            )
            # Cache high-confidence results
            if result and isinstance(result, dict):
                conf = result.get('confidence_score', 0)
                if conf and conf >= config.CONFIDENCE_THRESHOLD:
                    cache[cache_key] = (result, candidates, retrieval_meta)
            result = (result, candidates, retrieval_meta)

        # Unpack (now 3-tuple: result, candidates, retrieval_meta)
        if isinstance(result, (tuple, list)):
            mapping_result = result[0]
            candidates = result[1] if len(result) > 1 else []
            retrieval_meta = result[2] if len(result) > 2 else None
        else:
            mapping_result = result
            candidates = []
            retrieval_meta = None

        # Handle list edge case
        if isinstance(mapping_result, list):
            mapping_result = mapping_result[0] if mapping_result else {}

        # Extract prediction info
        pred_id = mapping_result.get("id") if isinstance(mapping_result, dict) else None
        pred_label = mapping_result.get("label") if isinstance(mapping_result, dict) else None

        # NEW: Find which retrieval source(s) contributed this prediction
        predicted_sources = None
        if pred_id and candidates:
            for cand in candidates:
                if cand.get("id") == pred_id:
                    predicted_sources = cand.get("retrieval_sources", [cand.get("source")])
                    if isinstance(predicted_sources, str):
                        predicted_sources = [predicted_sources]
                    break

        # Normalize ID if needed
        if normalize_id and pred_id:
            pred_id = normalize_id(pred_id)

        is_correct = pred_id in gold_ids if pred_id else False

        # Extract candidate info for analysis
        candidate_ids = [c.get("id") for c in candidates[:10]] if candidates else []
        candidate_labels = [c.get("label") for c in candidates[:5]] if candidates else []

        # Check if gold was in candidates (retrieval success)
        gold_in_candidates = bool(gold_ids & set(candidate_ids)) if candidate_ids else False

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - start_time

        return {
            # Core fields
            "query": query,
            "gold_ids": list(gold_ids),
            "predicted_id": pred_id,
            "predicted_label": pred_label,
            "predicted_sources": predicted_sources,  # NEW: Track which retrieval sources contributed
            "is_correct": is_correct,
            "confidence": mapping_result.get("confidence_score") if isinstance(mapping_result, dict) else None,

            # Retrieval analysis
            "candidate_count": len(candidates),
            "candidate_labels": candidate_labels,
            "gold_in_candidates": gold_in_candidates,

            # Per-attempt retrieval tracking
            "gold_first_found_at_attempt": retrieval_meta.get("gold_first_found_at") if retrieval_meta else None,
            "total_retrieval_attempts": retrieval_meta.get("total_attempts", 1) if retrieval_meta else 1,

            # Explanations (useful for understanding failures)
            "selector_explanation": mapping_result.get("selector_explanation", "")[:200] if isinstance(mapping_result, dict) else "",
            "scorer_explanation": mapping_result.get("scorer_explanation", "")[:200] if isinstance(mapping_result, dict) else "",

            # Performance metrics
            "time_seconds": round(elapsed_time, 3),
            "from_cache": from_cache,
        }
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        logger.error(f"Error evaluating '{item.get('query')}': {e}")
        return {
            "query": item.get("query", "unknown"),
            "gold_ids": list(item.get("gold_ids", [])),
            "predicted_id": None,
            "predicted_label": None,
            "is_correct": False,
            "confidence": None,
            "candidate_count": 0,
            "candidate_labels": [],
            "gold_in_candidates": False,
            "gold_first_found_at_attempt": None,
            "total_retrieval_attempts": 0,
            "selector_explanation": "",
            "scorer_explanation": "",
            "time_seconds": round(elapsed_time, 3),
            "from_cache": False,
            "error": str(e),
        }




async def evaluate_dataset(
    dataset_key: str,
    provider: str,
    sample_size: Optional[int],
    seed: int,
    output_dir: Path,
    no_cache: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single dataset with resume capability."""
    ds = DATASETS[dataset_key]

    print(f"\n{'='*60}")
    print(f"EVALUATING: {ds['name']} ({provider})")
    print(f"{'='*60}")

    # Configure for this dataset
    configure_for_dataset(dataset_key)

    # Load and sample data
    raw_items = load_dataset_raw(ds["input_file"])
    sampled_raw = sample_items(raw_items, sample_size, seed)
    items = [ds["load_item"](item, idx) for idx, item in enumerate(sampled_raw)]
    total_items = len(items)

    if sample_size is not None:
        print(f"Sampled {total_items} items (seed={seed})")
    else:
        print(f"Using full dataset: {total_items} items")

    # --- RESUME LOGIC ---
    results_jsonl_path = output_dir / "results.jsonl"
    processed_ids = set()
    completed_results = []

    if results_jsonl_path.exists():
        print(f"Found existing results at {results_jsonl_path}. Loading...")
        with results_jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    res = json.loads(line)
                    # Use query as the unique key for resumption
                    # (since we don't always have a guaranteed unique ID in input)
                    if res.get("dataset") == dataset_key:
                        processed_ids.add(res["query"])
                        completed_results.append(res)
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(processed_ids)} items already processed.")

    # Filter items
    items_to_process = [item for item in items if item["query"] not in processed_ids]
    print(f"Items remaining to process: {len(items_to_process)}")

    if not items_to_process:
        print("All items already processed for this dataset.")
        # If checks pass, we can just return the aggregation of completed_results
        valid_results = completed_results
    else:
        # Cache setup
        model_suffix = get_model_file_suffix()
        cache_path = ds["ontology_dir"] / f"cache_{model_suffix}.json"
        cache = load_cache(cache_path) if not no_cache and cache_path.exists() else {}

        # Create pipeline
        pipeline = create_pipeline(provider)
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

        try:
            # Prepare file for appending
            with results_jsonl_path.open('a', encoding='utf-8') as f_out:

                # Evaluate
                start_eval_time = time.perf_counter()
                normalize_id = ds.get("normalize_id")

                # We need to process and write as we go.
                # Since asyncio.gather waits for all, we can't easily write
                # strictly strictly sequentially without an intermediate callback
                # or modifying the task to write itself.
                # Let's modify the task wrapper to handle writing.

                async def process_and_save(item):
                    res = await evaluate_item(pipeline, item, semaphore, cache, ds["ontology_key"], normalize_id)
                    # Add dataset key for context in the combined file
                    res["dataset"] = dataset_key

                    # Write in a separate thread to ensure we don't block the event loop
                    # even for a moment (e.g. if disk is slow or flush hangs)
                    def _write_sync():
                        json_str = json.dumps(res)
                        f_out.write(json_str + "\n")
                        f_out.flush()

                    await asyncio.to_thread(_write_sync)
                    return res

                tasks = [process_and_save(item) for item in items_to_process]

                print(f"Running {len(tasks)} evaluations...")
                new_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect new valid results
                for i, r in enumerate(new_results):
                    if isinstance(r, Exception):
                        logger.error(f"Task {i} failed: {r}")
                        # We don't append to completed_results if it failed (it wasn't written)
                    else:
                        completed_results.append(r)

        finally:
            if pipeline:
                pipeline.close()
            if not no_cache:
                save_cache(cache_path, cache)

    # --- METRICS CALCULATION (on ALL results, old + new) ---
    valid_results = completed_results
    total = len(valid_results)

    # Recalculate timing roughly or just use current run?
    # For a resumed run, total wall clock is tricky. We'll just report on the *total data*.

    correct = sum(1 for r in valid_results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Retrieval analysis
    retrieval_success = sum(1 for r in valid_results if r.get("gold_in_candidates"))
    retrieval_rate = retrieval_success / total if total > 0 else 0.0

    # Confidence distribution
    confidences = [r.get("confidence") for r in valid_results if r.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    high_conf_count = sum(1 for c in confidences if c >= config.CONFIDENCE_THRESHOLD)

    # High confidence precision
    high_conf_results = [r for r in valid_results if (r.get("confidence") or 0) >= config.CONFIDENCE_THRESHOLD]
    high_conf_correct = sum(1 for r in high_conf_results if r["is_correct"])
    high_conf_precision = high_conf_correct / len(high_conf_results) if high_conf_results else 0.0

    # Error analysis
    error_count = sum(1 for r in valid_results if r.get("error"))
    no_prediction_count = sum(1 for r in valid_results if r.get("predicted_id") is None and not r.get("error"))
    cache_hits = sum(1 for r in valid_results if r.get("from_cache"))

    print(f"Results: {correct}/{total} = {accuracy:.2%} (Total Processed: {total})")

    return {
        "dataset": dataset_key,
        "dataset_name": ds["name"],
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "retrieval_success_count": retrieval_success,
            "retrieval_success_rate": round(retrieval_rate, 4),

            # Per-attempt retrieval success rates
            **{f"retrieval_success_rate@{k}": round(
                sum(1 for r in valid_results
                    if r.get("gold_first_found_at_attempt") is not None
                    and r.get("gold_first_found_at_attempt") <= k) / total, 4
            ) if total > 0 else 0.0 for k in range(1, config.MAX_PIPELINE_LOOPS + 1)},

            "avg_confidence": round(avg_confidence, 4),
            "high_confidence_count": high_conf_count,
            "high_confidence_precision": round(high_conf_precision, 4),
            "error_count": error_count,
            "no_prediction_count": no_prediction_count,
            "cache_hits": cache_hits,
            "concurrent_requests": config.MAX_CONCURRENT_REQUESTS,

            # NEW: Source attribution analysis (per-dataset)
            "source_attribution": calculate_source_metrics(valid_results),
        },
        "results": valid_results,
    }


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Run benchmark across all datasets")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed for sampling (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per dataset (default: use full dataset)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--provider", type=str, default=config.PIPELINE,
                        choices=["gemini", "vllm", "ollama"],
                        help="LLM provider to use")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest run for this model (matches *_{model_suffix})")
    args = parser.parse_args()

    # Configure
    config.PIPELINE = args.provider
    config.LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    setup_run_logging("run_benchmark")

    model_suffix = get_model_file_suffix()

    # --- DIRECTORY SETUP ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = None

    if args.resume:
        run_dir = find_latest_run_dir(model_suffix)
        if run_dir:
            print(f"RESUMING run from: {run_dir}")
        else:
            print(f"No existing run found for model '{model_suffix}' to resume.")

    if not run_dir:
        # Create new run directory
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"{run_timestamp}_{model_suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"STARTING NEW run in: {run_dir}")
    else:
        # If resuming, use the folder's timestamp for consistency in naming, or update?
        # Use folder name to extract timestamp if needed, but for now we just write to it.
        pass

    print(f"\n{'#'*60}")
    print(f"# BENCHMARK: {model_suffix}")
    print(f"# Output Directory: {run_dir}")
    print(f"# Datasets: {', '.join(args.datasets)}")
    if args.limit:
        print(f"# Samples per dataset: {args.limit} (seed={args.seed})")
    else:
        print(f"# Using full datasets")
    print(f"{'#'*60}")

    # Run all datasets
    all_results = {}
    summary = {
        "model": model_suffix,
        "provider": args.provider,
        "sample_size": args.limit,
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
    }

    for dataset_key in args.datasets:
        try:
            result = await evaluate_dataset(
                dataset_key=dataset_key,
                provider=args.provider,
                sample_size=args.limit,
                seed=args.seed,
                output_dir=run_dir,  # Pass result dir
                no_cache=args.no_cache,
            )
            all_results[dataset_key] = result
            summary["datasets"][dataset_key] = result["metrics"]
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_key}: {e}", exc_info=True)
            summary["datasets"][dataset_key] = {"error": str(e)}

    # Summary file
    summary_file = run_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Detail file
    details_file = run_dir / "details.json"
    with details_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Details saved: {details_file}")

    # Update comparison file (accumulates across models)
    comparison_file = RESULTS_DIR / "comparison.json"
    if comparison_file.exists():
        with comparison_file.open("r") as f:
            comparison = json.load(f)
    else:
        comparison = {"models": {}}

    # Use run_dir name as the key (it contains timestamp)
    run_key = run_dir.name
    comparison["models"][run_key] = {
        "provider": args.provider,
        "model": model_suffix,
        "timestamp": summary["timestamp"],
        "run_dir": str(run_dir.relative_to(RESULTS_DIR)),
        "sample_size": args.limit,
        "datasets": summary["datasets"],
    }
    comparison["last_updated"] = datetime.now().isoformat()

    with comparison_file.open("w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison updated: {comparison_file}")

    # Final summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {model_suffix}")
    for ds_key, metrics in summary["datasets"].items():
        if "error" in metrics:
            print(f"  {ds_key}: ERROR - {metrics['error']}")
        else:
            print(f"  {ds_key}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2%}")

    logging.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
