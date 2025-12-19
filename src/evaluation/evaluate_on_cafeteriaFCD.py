# src/evaluation/evaluate_pipeline.py
import xml.etree.ElementTree as ET
import logging
import json
import asyncio
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import os
from tqdm.asyncio import tqdm_asyncio
import argparse
import time
import statistics

from src.pipeline import create_pipeline
from src.pipeline import RAGPipeline
from src.pipeline_verbose import run_pipeline_verbose
from src import config
from src.utils.ontology_utils import uri_to_curie
from src.utils.cache import load_cache, save_cache
from src.utils.logging_config import setup_run_logging
from src.utils.token_tracker import token_tracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ----------------------------------------------------

# --- Configuration using pathlib ---
EVALUATION_XML_FILE = PROJECT_ROOT / "data" / "CafeteriaFCD_foodon_unique.xml"
# Output directory for CafeteriaFCD evaluation artifacts
CAFETERIA_RESULTS_DIR = PROJECT_ROOT / "cafeteria_results"

# --- Get a logger instance ---
# We no longer need basicConfig, as setup_run_logging will handle it.
logger = logging.getLogger(__name__)

# --- Re-used from original script, but updated for pathlib ---
def parse_evaluation_xml(xml_file_path: Path) -> list:
    """
    Parses the evaluation XML file to extract entities and their ground truth semantic tags.
    """
    if not xml_file_path.exists():
        logger.error(f"Evaluation XML file not found: {xml_file_path}")
        return []

    gold_standard_data = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for doc_idx, document_node in enumerate(root.findall('.//document')):
            for ann_idx, annotation_node in enumerate(document_node.findall('annotation')):
                entity_text_node = annotation_node.find('text')
                semantic_tags_node = annotation_node.find('infon[@key="semantic_tags"]')

                if entity_text_node is not None and semantic_tags_node is not None and entity_text_node.text and semantic_tags_node.text:
                    entity_text = entity_text_node.text.strip()
                    raw_tags = semantic_tags_node.text.strip()
                    true_uris = {tag.strip() for tag in raw_tags.split(';') if tag.strip()}
                    true_curies = {uri_to_curie(uri, config.CURIE_PREFIX_MAP) for uri in true_uris} - {None}

                    if entity_text and true_curies:
                        gold_standard_data.append({
                            'text': entity_text,
                            'true_curies': list(true_curies), # Convert to list for JSON
                        })
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []

    logger.info(f"Successfully parsed {len(gold_standard_data)} entities from {xml_file_path}")
    return gold_standard_data

# --- REFACTORED ASYNC EVALUATION LOGIC (Unchanged from previous step) ---
async def evaluate_full_pipeline(
    pipeline: RAGPipeline,
    gold_standard_data: list,
    cache: dict,
    semaphore: asyncio.Semaphore,
    *,
    no_cache: bool = False,
    lexical_k: Optional[int] = None,
    vector_k: Optional[int] = None,
    retrieval_diag: bool = False,
) -> Tuple[float, int, int, int, int, List[Dict], Dict]:
    """
    Evaluates the full RAG pipeline asynchronously, using a cache.
    """
    total_entities = len(gold_standard_data)
    if total_entities == 0:
        return 0.0, 0, 0, 0, 0, []

    hits, cache_hits, failures = 0, 0, 0
    misses = []
    tasks_to_run = []
    queries_for_tasks = []

    # Metrics accumulators (only for items we actively run, not cache hits)
    first_hits = 0
    final_hits = 0
    attempted = 0
    retries = 0
    synonyms_used_count = 0
    latencies: List[float] = []
    conf_correct_pairs: List[Tuple[float, int]] = []  # (confidence, correct)

    logger.info("Checking cache and scheduling pipeline runs for misses...")
    # helper to time each run
    async def _run_one(q: str):
        # Limit end-to-end pipeline concurrency to respect quotas
        async with semaphore:
            t0 = time.perf_counter()
            triple = None
            try:
                triple = await run_pipeline_verbose(
                    pipeline,
                    q,
                    lexical_k=lexical_k or config.DEFAULT_K_LEXICAL,
                    vector_k=vector_k or config.DEFAULT_K_VECTOR,
                )
            except Exception as e:
                logger.error("Verbose pipeline failed for query '%s': %s", q, e, exc_info=True)
            dt = time.perf_counter() - t0
            return q, triple, dt

    for item in gold_standard_data:
        query = item['text']
        if (not no_cache) and (query in cache):
            cache_hits += 1
        else:
            task = _run_one(query)
            tasks_to_run.append(task)
            queries_for_tasks.append(query)
    
    logger.info(f"Found {cache_hits} items in cache. Running pipeline for {len(tasks_to_run)} new items.")

    pipeline_results = []
    if tasks_to_run:
        t0 = time.perf_counter()
        pipeline_results = await tqdm_asyncio.gather(*tasks_to_run, desc="Evaluating Pipeline")
        eval_time = time.perf_counter() - t0
    
    for i, packed in enumerate(pipeline_results):
        query = queries_for_tasks[i]
        _, triple, dt = packed
        # triple = (final_result, candidates, history)
        if triple and triple[0]:
            final_result, candidates, history = triple
            # Cache only the tuple (final_result, candidates) for compatibility
            cache[query] = (final_result, candidates)
            
            # Metrics from verbose run
            attempted += 1
            iter_count = len(history) if history else 0
            if iter_count > 1:
                retries += 1
            if history and any(step.get('synonyms') for step in history if isinstance(step.get('synonyms'), list)):
                synonyms_used_count += 1
            latencies.append(dt)

            first_id = None
            if history and isinstance(history[0], dict):
                first_res = history[0].get('result') or {}
                first_id = first_res.get('id')

            true_curies = set(next(item['true_curies'] for item in gold_standard_data if item['text'] == query))
            is_first_hit = bool(first_id and (first_id in true_curies))
            is_final_hit = bool(final_result and final_result.get('id') in true_curies)
            if is_first_hit:
                first_hits += 1
            if is_final_hit:
                final_hits += 1

            conf = float(final_result.get('confidence_score', 0.0)) if final_result else 0.0
            conf_correct_pairs.append((conf, 1 if is_final_hit else 0))

    for item in gold_standard_data:
        query = item['text']
        true_curies = set(item['true_curies'])
        result_tuple = cache.get(query)

        if not result_tuple or not result_tuple[0]:
            logger.warning(f"Failure: Pipeline returned no result for '{query}'.")
            failures += 1
            continue

        final_result, candidates = result_tuple
        chosen_curie = final_result.get('id')

        if chosen_curie in true_curies:
            hits += 1
        else:
            logger.debug(f"MISS! Query: '{query}'. Chosen: '{chosen_curie}', Expected: {true_curies}.")
            misses.append({
                "query": query,
                "chosen_curie": chosen_curie,
                "true_curies": list(true_curies),
                # Capture both selector and scorer explanations from the pipeline
                "selector_explanation": final_result.get("selector_explanation", "N/A"),
                "scorer_explanation": final_result.get("scorer_explanation", "N/A"),
                "confidence_score": final_result.get("confidence_score", 0.0),
                "candidates_provided": [c.get('id') for c in candidates if c.get('id')]
            })

    valid_attempts = total_entities - failures
    accuracy = hits / valid_attempts if valid_attempts > 0 else 0.0

    # Compute first/final accuracies for actively attempted items (excluding cache hits)
    acc_first = first_hits / attempted if attempted > 0 else 0.0
    acc_final_attempts = final_hits / attempted if attempted > 0 else 0.0
    retry_rate = retries / attempted if attempted > 0 else 0.0
    synonyms_rate = synonyms_used_count / attempted if attempted > 0 else 0.0

    # Latency stats
    lat_stats = {
        "count": len(latencies),
        "avg": float(statistics.mean(latencies)) if latencies else 0.0,
        "p50": float(statistics.median(latencies)) if latencies else 0.0,
        "p95": float(sorted(latencies)[int(0.95 * (len(latencies)-1))]) if latencies else 0.0,
    }

    # ECE over 10 bins
    ece = 0.0
    if conf_correct_pairs:
        bins = [[] for _ in range(10)]
        for conf, corr in conf_correct_pairs:
            idx = min(9, max(0, int(conf * 10)))
            bins[idx].append((conf, corr))
        N = len(conf_correct_pairs)
        for b in bins:
            if not b:
                continue
            avg_conf = sum(c for c, _ in b) / len(b)
            acc_bin = sum(corr for _, corr in b) / len(b)
            ece += (len(b) / N) * abs(avg_conf - acc_bin)

    summary = {
        "overall": {
            "total": total_entities,
            "failures": failures,
            "cache_hits": cache_hits,
            "valid_attempts": valid_attempts,
            "accuracy_overall": accuracy,
        },
        "attempted_only": {
            "attempted": attempted,
            "acc_first": acc_first,
            "acc_final": acc_final_attempts,
            "retry_rate": retry_rate,
            "synonyms_rate": synonyms_rate,
            "ece_10bin": ece,
            "latency": lat_stats,
        },
        "tokens": {
            "total": token_tracker.get_total_usage(),
            "by_model": {
                model: {
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "calls": dict(token_tracker.call_counts.get(model, {})),
                }
                for model, data in token_tracker.counts.items()
            },
        },
    }

    return accuracy, total_entities, hits, cache_hits, failures, misses, summary

# --- REFACTORED ASYNC MAIN FUNCTION ---
async def main():
    # --- NEW: Set up run-specific logging ---
    # This will create a unique log file in the `logs/` directory
    setup_run_logging("evaluation_run")
    # ----------------------------------------
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Run without reading or writing the persistent cache",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Number of items to evaluate (0 = all). Default: 10",
    )
    parser.add_argument(
        "--lexical-k",
        type=int,
        default=None,
        help="Override lexical candidate fanout (default from config)",
    )
    parser.add_argument(
        "--vector-k",
        type=int,
        default=None,
        help="Override vector/semantic candidate fanout (default from config)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override confidence threshold τ used for retries (default from config)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override max concurrent pipeline runs (default from config)",
    )
    args, _ = parser.parse_known_args()
    no_cache = bool(args.no_cache)
    limit = int(args.limit) if args.limit is not None else 10
    # Resolve effective K values and threshold
    effective_lex_k = args.lexical_k if args.lexical_k is not None else int(config.DEFAULT_K_LEXICAL)
    effective_vec_k = args.vector_k if args.vector_k is not None else int(config.DEFAULT_K_VECTOR)
    effective_tau = float(args.tau) if args.tau is not None else float(config.CONFIDENCE_THRESHOLD)
    
    # If user overrides tau, patch config for this run so pipeline uses it
    if args.tau is not None:
        logging.getLogger(__name__).info(f"Overriding confidence threshold: {config.CONFIDENCE_THRESHOLD} -> {effective_tau}")
        config.CONFIDENCE_THRESHOLD = effective_tau

    logger.info("Starting Full Pipeline Evaluation Script...")
    logger.info(f"Evaluating Pipeline: '{config.PIPELINE}'")
    if no_cache:
        logger.warning("Cache disabled: ignoring cache and not persisting it.")

    gold_standard_data = parse_evaluation_xml(EVALUATION_XML_FILE)
    if not gold_standard_data:
        logger.error("Failed to load or parse gold standard data. Exiting.")
        return
    # Apply sampling/limit for quick testing
    if limit and limit > 0:
        gold_standard_data = gold_standard_data[:limit]
        logger.warning(f"Processing limited to {limit} items for this run.")

    pipeline = None
    cache = {} if no_cache else load_cache(config.PIPELINE_CACHE_PATH)
    
    try:
        logger.info(f"Initializing RAG pipeline: '{config.PIPELINE}'...")
        pipeline = create_pipeline(config.PIPELINE)
        logger.info("Pipeline initialized successfully.")

        max_conc = int(args.max_concurrency) if args.max_concurrency is not None else int(config.MAX_CONCURRENT_REQUESTS)
        logger.info(f"Using max concurrency: {max_conc}")
        semaphore = asyncio.Semaphore(max_conc)

        logger.info(f"Starting evaluation for {len(gold_standard_data)} entities...")
        accuracy, total, correct, cache_hits, failures, misses, summary = await evaluate_full_pipeline(
            pipeline,
            gold_standard_data,
            cache,
            semaphore,
            no_cache=no_cache,
            lexical_k=effective_lex_k,
            vector_k=effective_vec_k,
            retrieval_diag=False,
        )

        logger.info("--- Evaluation Complete ---")
        logger.info(f"Total entities evaluated: {total}")
        logger.info(f"Cache Hits: {cache_hits}")
        logger.info(f"Pipeline Runs (Cache Misses): {total - cache_hits}")
        logger.info(f"Pipeline Failures (errors or no result): {failures}")
        logger.info("-" * 27)

        valid_attempts = total - failures
        logger.info(f"Valid attempts for selector: {valid_attempts}")
        logger.info(f"Correct selections (Hits): {correct}")

        if valid_attempts > 0:
            logger.info(f"Accuracy (overall): {accuracy:.4f} ({correct}/{valid_attempts})")
        else:
            logger.info("Accuracy: N/A (no valid attempts were made)")

        # Report easy-win metrics
        att = summary["attempted_only"]
        logger.info(f"First vs Final Acc@1 on attempted: {att['acc_first']:.4f} -> {att['acc_final']:.4f}")
        logger.info(f"Retry coverage: {att['retry_rate']:.2%}; Synonyms usage: {att['synonyms_rate']:.2%}")
        logger.info(f"Latency (s) — avg: {att['latency']['avg']:.3f}, p50: {att['latency']['p50']:.3f}, p95: {att['latency']['p95']:.3f}")
        logger.info(f"Calibration ECE (10 bins): {att['ece_10bin']:.4f}")

        # Ensure output directory exists
        CAFETERIA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Construct parameterized filenames
        def _fmt_tau(x: float) -> str:
            s = ("%.3f" % x).rstrip('0').rstrip('.')
            return s
        fname_suffix = f"klex{effective_lex_k}_ksem{effective_vec_k}_tau{_fmt_tau(effective_tau)}"

        misses_path = CAFETERIA_RESULTS_DIR / f"cafeteria_misses_{config.PIPELINE}_{fname_suffix}.json"
        logger.info(f"Saving {len(misses)} incorrect selections to {misses_path}")
        with open(misses_path, 'w', encoding='utf-8') as f:
            json.dump(misses, f, indent=4)

        # Also save a compact summary next to misses
        summary_path = CAFETERIA_RESULTS_DIR / f"cafeteria_summary_{config.PIPELINE}_{fname_suffix}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved summary metrics to {summary_path}")

        logger.info("Evaluation finished.")

    except Exception as e:
        logger.error(f"Failed to initialize or run the pipeline: {e}", exc_info=True)
    finally:
        # --- NEW: Log and print token usage ---
        token_report = token_tracker.report_usage()
        logger.info(token_report)
        # --------------------------------------

        if pipeline:
            logger.info("Closing pipeline resources...")
            pipeline.close()
        # Persist cache only if enabled
        if not no_cache:
            save_cache(config.PIPELINE_CACHE_PATH, cache)

if __name__ == "__main__":
    if not EVALUATION_XML_FILE.exists():
        # Use logger here as basicConfig might not be set up if the file doesn't exist
        logging.error(f"Evaluation XML file '{EVALUATION_XML_FILE}' not found.")
    else:
        asyncio.run(main())
