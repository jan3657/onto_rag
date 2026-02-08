# src/evaluation/evaluate_nlm_gene.py
"""
Evaluate the RAG pipeline on the NLM Gene entity linking dataset.

The dataset links gene mentions to NCBI Gene IDs (NCBIGene:XXXXX).
The ontology source is gene_info.tsv from NCBI.

Usage:
  python -m src.evaluation.evaluate_nlm_gene --ingest --tax-ids 9606,10090  # Build indexes (first run)
  python -m src.evaluation.evaluate_nlm_gene --limit 2  # Test with 2 entities
  python -m src.evaluation.evaluate_nlm_gene            # Full evaluation
"""

import argparse
import asyncio
import gzip
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src import config
from src.pipeline import create_pipeline
from src.utils.logging_config import setup_run_logging
from src.utils.cache import load_cache, save_cache
from src.evaluation.source_metrics import (
    calculate_source_metrics,
    calculate_retrieval_source_metrics,
)

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
        # MiniLM embeddings (backward compatible)
        'embeddings_path': ONTOLOGY_DIR / "embeddings.json",
        'embeddings_minilm_path': ONTOLOGY_DIR / "embeddings_minilm.json",
        'embeddings_sapbert_path': ONTOLOGY_DIR / "embeddings_sapbert.json",
        'whoosh_index_dir': ONTOLOGY_DIR / "whoosh_index",
        # MiniLM FAISS (backward compatible)
        'faiss_index_path': ONTOLOGY_DIR / "faiss_index.bin",
        'faiss_metadata_path': ONTOLOGY_DIR / "faiss_metadata.json",
        # Explicit MiniLM and SapBERT FAISS paths
        'faiss_index_minilm_path': ONTOLOGY_DIR / "faiss_index_minilm.bin",
        'faiss_metadata_minilm_path': ONTOLOGY_DIR / "faiss_metadata_minilm.json",
        'faiss_index_sapbert_path': ONTOLOGY_DIR / "faiss_index_sapbert.bin",
        'faiss_metadata_sapbert_path': ONTOLOGY_DIR / "faiss_metadata_sapbert.json",
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
# OUTPUT_FILE and CACHE_PATH are now dynamic based on model - see main()

# ============================================================
# INGESTION CONFIGURATION
# ============================================================

# Filter to specific organism(s) by tax_id
# Mouse = 10090, Human = 9606, Rat = 10116
# Set to None to include all organisms (WARNING: 65M+ entries!)
FILTER_TAX_IDS = [9606, 10090]  # Human + mouse (default for mixed NLM Gene mentions)

# ============================================================
# LOGGING SETUP
# ============================================================

# Default log level (can be overridden by --debug CLI flag)
# This will be set after argument parsing
logger = logging.getLogger(__name__)


# ============================================================
# INGESTION
# ============================================================

def parse_tax_ids_arg(raw_tax_ids: Optional[str]) -> Optional[List[int]]:
    """
    Parse a comma-separated list of tax IDs.

    Special values:
    - "all" / "none" / "*" => None (no filtering)
    """
    if raw_tax_ids is None:
        return FILTER_TAX_IDS

    value = raw_tax_ids.strip()
    if value.lower() in {"all", "none", "*"}:
        return None

    parsed: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise ValueError(
                f"Invalid tax ID '{part}'. Use comma-separated integers, e.g. 9606,10090"
            ) from exc

    if not parsed:
        raise ValueError("No valid tax IDs provided. Use values like 9606,10090 or 'all'.")
    return parsed

def run_ingestion(max_rows: Optional[int] = None) -> None:
    """
    Build indexes for NCBI Gene with both MiniLM and SapBERT embeddings.

    Uses the generic TSV parser with gene_info.tsv-specific column mappings.
    """
    from src.ingestion import parse_tsv, build_whoosh_index, build_embeddings, build_faiss_index

    cfg = config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]

    print(f"\n{'='*60}")
    print(f"INGESTING: NCBI Gene (gene_info.tsv) with DUAL-VECTOR")
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

    print(f"\n[1/6] Parsing gene_info.tsv...")
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

    print(f"\n[2/6] Building Whoosh index...")
    build_whoosh_index(cfg['dump_json_path'], cfg['whoosh_index_dir'])

    print(f"\n[3/6] Building MiniLM embeddings...")
    build_embeddings(
        cfg['dump_json_path'],
        cfg['embeddings_minilm_path'],
        model_name=config.EMBEDDING_MODEL_NAME
    )

    print(f"\n[4/6] Building MiniLM FAISS index...")
    build_faiss_index(
        cfg['embeddings_minilm_path'],
        cfg['faiss_index_minilm_path'],
        cfg['faiss_metadata_minilm_path']
    )

    print(f"\n[5/6] Building SapBERT embeddings (biomedical, will take longer)...")
    build_embeddings(
        cfg['dump_json_path'],
        cfg['embeddings_sapbert_path'],
        model_name=config.SAPBERT_MODEL_NAME
    )

    print(f"\n[6/6] Building SapBERT FAISS index...")
    build_faiss_index(
        cfg['embeddings_sapbert_path'],
        cfg['faiss_index_sapbert_path'],
        cfg['faiss_metadata_sapbert_path']
    )

    print(f"\n{'='*60}")
    print("✅ DUAL-VECTOR INGESTION COMPLETE!")
    print(f"{'='*60}")


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(input_path: Path, limit: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load NLM Gene test dataset (JSONL.gz format).

    NOTE: Uses random sampling when limit is specified to match run_benchmark.py behavior.
    Set seed for reproducibility.
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

    # Random sampling if limit specified (matching run_benchmark.py behavior)
    if limit and len(items) > limit:
        random.seed(seed)
        items = random.sample(items, limit)

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
    ontology_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a single mention using the pipeline."""
    start_time = time.perf_counter()
    query = item["query"]
    context = item.get("context", "")
    item_id = item["id"]
    gold_ids = set(item.get("gold_ids", []))
    gold_ids_in_ontology = sorted(g for g in gold_ids if g in ontology_ids) if ontology_ids is not None else []
    gold_covered_by_index = bool(gold_ids_in_ontology) if ontology_ids is not None else None

    # Check cache
    cache_key = f"{query}|||{context[:100]}" if context else query
    from_cache = False
    if cache_key in cache:
        logger.debug(f"Cache hit: {query}")
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
            target_ontologies=[ONTOLOGY_KEY],
            gold_ids=gold_ids,
        )
        # Only cache high-confidence results
        if result and isinstance(result, dict):
            conf = result.get('confidence_score', 0)
            if conf and conf >= config.CONFIDENCE_THRESHOLD:
                cache[cache_key] = (result, candidates, retrieval_meta)
        result = (result, candidates, retrieval_meta)

    # Unpack result (now 3-tuple)
    if isinstance(result, tuple):
        mapping_result = result[0]
        candidates = result[1] if len(result) > 1 else []
        retrieval_meta = result[2] if len(result) > 2 else None
    else:
        mapping_result, candidates, retrieval_meta = result, [], None

    # Extract predicted ID
    pred_id = normalize_gene_id(mapping_result.get("id")) if isinstance(mapping_result, dict) else None

    # NEW: Find which retrieval source(s) contributed this prediction
    predicted_sources = None
    if pred_id and candidates:
        # Denormalize to match candidates
        search_id = pred_id
        for cand in candidates:
            if cand.get("id") == search_id:
                predicted_sources = cand.get("retrieval_sources", [cand.get("source")])
                if isinstance(predicted_sources, str):
                    predicted_sources = [predicted_sources]
                break

    # Check if correct
    is_correct = pred_id in gold_ids if pred_id else False
    elapsed_time = time.perf_counter() - start_time
    candidate_ids = [c.get("id") for c in candidates[:10]] if candidates else []
    gold_in_candidates_final_top10 = bool(gold_ids & set(candidate_ids)) if candidate_ids else False
    gold_found_any_attempt = (
        retrieval_meta.get("gold_first_found_at") is not None
        if retrieval_meta else gold_in_candidates_final_top10
    )
    gold_found_by_source_any_attempt = {
        "lexical": False,
        "minilm": False,
        "sapbert": False,
        "merged_top10": False,
        "merged_any_rank": False,
    }
    if retrieval_meta:
        for attempt in retrieval_meta.get("retrieval_history", []):
            by_source = attempt.get("gold_found_by_source", {}) or {}
            for source_name in gold_found_by_source_any_attempt:
                gold_found_by_source_any_attempt[source_name] = (
                    gold_found_by_source_any_attempt[source_name] or bool(by_source.get(source_name))
                )

    simplified = simplify_mapping_result(mapping_result)

    return {
        "id": item_id,
        "query": query,
        "context": context[:200] if context else "",
        "gold_ids": list(gold_ids),
        "predicted_id": pred_id,
        "predicted_sources": predicted_sources,  # NEW: Track retrieval sources
        "is_correct": is_correct,
        "confidence": mapping_result.get("confidence_score") if mapping_result else None,
        "mapping_result": simplified if simplified else "No mapping found",
        "candidates": [c.get("id") for c in candidates[:5]] if candidates else [],
        # Backward-compatible field: top-10 hit in the final returned candidate list.
        "gold_in_candidates": gold_in_candidates_final_top10,
        "gold_in_candidates_final_top10": gold_in_candidates_final_top10,
        # Any-rank hit across all retrieval attempts.
        "gold_found_any_attempt": gold_found_any_attempt,
        "gold_found_by_source_any_attempt": gold_found_by_source_any_attempt,
        "gold_covered_by_index": gold_covered_by_index,
        "gold_ids_in_ontology": gold_ids_in_ontology,
        "gold_first_found_at_attempt": retrieval_meta.get("gold_first_found_at") if retrieval_meta else None,
        "gold_first_found_at_attempt_top10": retrieval_meta.get("gold_first_found_at_top10") if retrieval_meta else None,
        "total_retrieval_attempts": retrieval_meta.get("total_attempts", 1) if retrieval_meta else 1,
        "time_seconds": round(elapsed_time, 3),
        "from_cache": from_cache,
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
        "--tax-ids",
        type=str,
        default="9606,10090",
        help="Comma-separated tax IDs for ingestion (e.g., 9606,10090). Use 'all' for no filtering.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling test items (default: 42)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=config.PIPELINE,
        choices=["gemini", "vllm", "ollama"],
        help="LLM provider to use (default: from config.PIPELINE)",
    )

    args = parser.parse_args()

    # Allow runtime override of ingestion species filter.
    global FILTER_TAX_IDS
    try:
        FILTER_TAX_IDS = parse_tax_ids_arg(args.tax_ids)
    except ValueError as exc:
        parser.error(str(exc))

    # Update config.PIPELINE to match CLI arg (affects logging and model naming)
    config.PIPELINE = args.provider

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
    print(f"EVALUATING: NLM Gene → NCBI Gene ({args.provider})")
    print(f"{'='*60}\n")

    # Build dynamic paths based on MODEL for separate results per model
    from src.utils.model_utils import get_model_file_suffix
    model_suffix = get_model_file_suffix()
    output_file = ONTOLOGY_DIR / f"results_{model_suffix}.json"
    cache_path = ONTOLOGY_DIR / f"cache_{model_suffix}.json"

    # Load cache
    cache: Dict[str, Any] = {}
    if not args.no_cache and cache_path.exists():
        cache = load_cache(cache_path)
        logger.info(f"Loaded {len(cache)} cached results")

    pipeline = None
    try:
        # Load dataset
        items = load_dataset(INPUT_FILE, limit=args.limit, seed=args.seed)
        print(f"Loaded {len(items)} test mentions")

        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = create_pipeline(args.provider)

        # Load ontology IDs once for coverage diagnostics.
        ontology_ids: Optional[Set[str]] = None
        dump_path = config.ONTOLOGIES_CONFIG[ONTOLOGY_KEY]["dump_json_path"]
        if dump_path.exists():
            with dump_path.open("r", encoding="utf-8") as f:
                ontology_ids = set(json.load(f).keys())
            logger.info(f"Loaded {len(ontology_ids)} ontology IDs for coverage analysis")
        else:
            logger.warning(f"Ontology dump not found for coverage analysis: {dump_path}")

        # Setup concurrency
        max_conc = args.max_concurrency or config.MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(max_conc)

        # Process items
        start_eval_time = time.perf_counter()
        tasks = [
            evaluate_item(pipeline, item, semaphore, cache, ontology_ids=ontology_ids)
            for item in items
        ]

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
        attempted_total = len(items)
        task_failed_count = attempted_total - total
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
        retrieval_success_any_attempt = sum(1 for r in results if r.get("gold_found_any_attempt"))
        retrieval_rate_any_attempt = retrieval_success_any_attempt / total if total > 0 else 0.0
        retrieval_success_final_top10 = sum(1 for r in results if r.get("gold_in_candidates_final_top10"))
        retrieval_rate_final_top10 = retrieval_success_final_top10 / total if total > 0 else 0.0

        # Confidence metrics
        confidences = [r.get("confidence") for r in results if r.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        high_conf_count = sum(1 for c in confidences if c >= config.CONFIDENCE_THRESHOLD)
        high_conf_results = [r for r in results if (r.get("confidence") or 0) >= config.CONFIDENCE_THRESHOLD]
        high_conf_correct = sum(1 for r in high_conf_results if r["is_correct"])
        high_conf_precision = high_conf_correct / len(high_conf_results) if high_conf_results else 0.0

        # Coverage metrics
        coverage_known = [r for r in results if r.get("gold_covered_by_index") is not None]
        covered_count = sum(1 for r in coverage_known if r.get("gold_covered_by_index"))
        covered_rate = covered_count / len(coverage_known) if coverage_known else 0.0
        covered_results = [r for r in results if r.get("gold_covered_by_index") is True]
        covered_total = len(covered_results)
        covered_accuracy = (
            sum(1 for r in covered_results if r["is_correct"]) / covered_total
            if covered_total > 0 else 0.0
        )
        covered_retrieval_any = (
            sum(1 for r in covered_results if r.get("gold_found_any_attempt")) / covered_total
            if covered_total > 0 else 0.0
        )
        covered_retrieval_final_top10 = (
            sum(1 for r in covered_results if r.get("gold_in_candidates_final_top10")) / covered_total
            if covered_total > 0 else 0.0
        )

        # Error counts
        row_error_count = sum(1 for r in results if r.get("error"))
        error_count = row_error_count + task_failed_count
        no_prediction_count = sum(1 for r in results if r.get("predicted_id") is None and not r.get("error"))
        cache_hits = sum(1 for r in results if r.get("from_cache"))

        metrics = {
            "attempted_total": attempted_total,
            "total": total,
            "task_failed_count": task_failed_count,
            "correct": correct,
            "accuracy": accuracy,
            "total_time_seconds": round(wall_clock_time, 2),
            "items_per_second": round(items_per_second, 2),
            "avg_latency_seconds": round(avg_latency, 3),
            "concurrent_requests": max_conc,
            # Primary retrieval metric: gold found in any rank at any attempt.
            "retrieval_success_count": retrieval_success_any_attempt,
            "retrieval_success_rate": round(retrieval_rate_any_attempt, 4),
            # Secondary retrieval metric: gold in final returned candidate top-10.
            "retrieval_success_count_final_top10": retrieval_success_final_top10,
            "retrieval_success_rate_final_top10": round(retrieval_rate_final_top10, 4),
            **{f"retrieval_success_rate@{k}": round(
                sum(1 for r in results
                    if r.get("gold_first_found_at_attempt") is not None
                    and r.get("gold_first_found_at_attempt") <= k) / total, 4
            ) if total > 0 else 0.0 for k in range(1, config.MAX_PIPELINE_LOOPS + 1)},
            **{f"retrieval_success_rate_top10@{k}": round(
                sum(1 for r in results
                    if r.get("gold_first_found_at_attempt_top10") is not None
                    and r.get("gold_first_found_at_attempt_top10") <= k) / total, 4
            ) if total > 0 else 0.0 for k in range(1, config.MAX_PIPELINE_LOOPS + 1)},
            "gold_covered_by_index_count": covered_count,
            "gold_coverage_rate": round(covered_rate, 4),
            "covered_subset_total": covered_total,
            "covered_subset_accuracy": round(covered_accuracy, 4),
            "covered_subset_retrieval_success_rate": round(covered_retrieval_any, 4),
            "covered_subset_retrieval_success_rate_final_top10": round(covered_retrieval_final_top10, 4),
            "avg_confidence": round(avg_confidence, 4),
            "high_confidence_count": high_conf_count,
            "high_confidence_precision": round(high_conf_precision, 4),
            "error_count": error_count,
            "no_prediction_count": no_prediction_count,
            "cache_hits": cache_hits,

            # NEW: Source attribution analysis
            "source_attribution": calculate_source_metrics(results),
            "retrieval_source_attribution": calculate_retrieval_source_metrics(results),
        }

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Attempted: {attempted_total}")
        print(f"Total:     {total}")
        print(f"Failed:    {task_failed_count}")
        print(f"Correct:  {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Gold coverage (in indexed ontology): {covered_count}/{total} ({covered_rate:.2%})")
        print(f"Retrieval success (any attempt, any rank): {retrieval_success_any_attempt}/{total} ({retrieval_rate_any_attempt:.2%})")
        print(f"Retrieval success (final top-10 only): {retrieval_success_final_top10}/{total} ({retrieval_rate_final_top10:.2%})")

        # Show sample results
        print(f"\n--- Sample Results ---")
        for r in results[:5]:
            status = "✅" if r["is_correct"] else "❌"
            print(f"{status} '{r['query']}' → {r['predicted_id']} (gold: {r['gold_ids']})")

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "results": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

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
            save_cache(cache_path, cache)
            logger.info(f"Saved {len(cache)} cached results")


if __name__ == "__main__":
    asyncio.run(main())
