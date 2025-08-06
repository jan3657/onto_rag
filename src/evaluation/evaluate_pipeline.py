# src/evaluation/evaluate_pipeline.py
import xml.etree.ElementTree as ET
import sys
import logging
import json
import asyncio
from pathlib import Path
from typing import Tuple, List, Dict
import os
from tqdm.asyncio import tqdm_asyncio


# Add project root to Python path to allow direct imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- MODIFIED IMPORTS ---
from src.pipeline.pipeline_factory import get_pipeline
from src.pipeline.base_pipeline import BaseRAGPipeline
from src import config
from src.utils.ontology_utils import uri_to_curie
from src.utils.cache import load_cache, save_cache
# --- NEW IMPORTS FOR LOGGING AND TOKEN TRACKING ---
from src.utils.logging_config import setup_run_logging
from src.utils.token_tracker import token_tracker
# ----------------------------------------------------

# --- Configuration using pathlib ---
EVALUATION_XML_FILE = PROJECT_ROOT / "data" / "CafeteriaFCD_foodon_unique.xml"
EVALUATION_OUTPUT_FILE = PROJECT_ROOT / f"evaluation_results_{config.PIPELINE}.json"

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
    pipeline: BaseRAGPipeline,
    gold_standard_data: list,
    cache: dict,
    semaphore: asyncio.Semaphore
) -> Tuple[float, int, int, int, int, List[Dict]]:
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

    logger.info("Checking cache and scheduling pipeline runs for misses...")
    for item in gold_standard_data:
        query = item['text']
        if query in cache:
            cache_hits += 1
        else:
            task = pipeline.run(query=query, semaphore=semaphore)
            tasks_to_run.append(task)
            queries_for_tasks.append(query)
    
    logger.info(f"Found {cache_hits} items in cache. Running pipeline for {len(tasks_to_run)} new items.")

    pipeline_results = []
    if tasks_to_run:
        pipeline_results = await tqdm_asyncio.gather(*tasks_to_run, desc="Evaluating Pipeline")
    
    for i, result_tuple in enumerate(pipeline_results):
        query = queries_for_tasks[i]
        if result_tuple and result_tuple[0]:
            cache[query] = result_tuple

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
                "explanation": final_result.get("explanation", "N/A"),
                "confidence_score": final_result.get("confidence_score", 0.0),
                "candidates_provided": [c.get('id') for c in candidates if c.get('id')]
            })

    valid_attempts = total_entities - failures
    accuracy = hits / valid_attempts if valid_attempts > 0 else 0.0
    return accuracy, total_entities, hits, cache_hits, failures, misses

# --- REFACTORED ASYNC MAIN FUNCTION ---
async def main():
    # --- NEW: Set up run-specific logging ---
    # This will create a unique log file in the `logs/` directory
    setup_run_logging("evaluation_run")
    # ----------------------------------------
    
    logger.info("Starting Full Pipeline Evaluation Script...")
    logger.info(f"Evaluating Pipeline: '{config.PIPELINE}'")

    gold_standard_data = parse_evaluation_xml(EVALUATION_XML_FILE)
    if not gold_standard_data:
        logger.error("Failed to load or parse gold standard data. Exiting.")
        return

    pipeline = None
    cache = load_cache(config.PIPELINE_CACHE_PATH)
    
    try:
        logger.info(f"Initializing RAG pipeline: '{config.PIPELINE}'...")
        pipeline = get_pipeline(config.PIPELINE)
        logger.info("Pipeline initialized successfully.")

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

        logger.info(f"Starting evaluation for {len(gold_standard_data)} entities...")
        accuracy, total, correct, cache_hits, failures, misses = await evaluate_full_pipeline(
            pipeline,
            gold_standard_data,
            cache,
            semaphore
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
            logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{valid_attempts})")
        else:
            logger.info("Accuracy: N/A (no valid attempts were made)")

        logger.info(f"Saving {len(misses)} incorrect selections to {EVALUATION_OUTPUT_FILE}")
        with open(EVALUATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(misses, f, indent=4)

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
        # Save the cache regardless of errors
        save_cache(config.PIPELINE_CACHE_PATH, cache)

if __name__ == "__main__":
    if not EVALUATION_XML_FILE.exists():
        # Use logger here as basicConfig might not be set up if the file doesn't exist
        logging.error(f"Evaluation XML file '{EVALUATION_XML_FILE}' not found.")
    else:
        asyncio.run(main())