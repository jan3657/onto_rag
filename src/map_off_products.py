#!/usr/bin/env python
"""
run_on_off.py – Map parsed ingredient lists to standardized ontology labels using OntoRag.

*** FINAL VERSION: Stores {'id', 'label'} progressively and safely. ***

Input  : your_data.json (product_id ➔ { "parsed_ingredients": list[str], ... })
Output : your_data_standardized.json (product_id ➔ { ..., "standardized_ingredients": list[dict|None] })

The script re‑uses the OntoRag RAG pipeline, leverages a persistent cache to avoid
re‑processing identical ingredient queries, and processes products concurrently with
an asyncio‑compatible semaphore governed by config.MAX_CONCURRENT_REQUESTS.

The script saves the cache after each product is processed to ensure resilience.
For high-confidence matches, it caches a dictionary containing the term's 'id' and 'label'.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm.asyncio import tqdm as asyncio_tqdm

from src.utils.logging_config import setup_run_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.adapters.pipeline_factory import create_pipeline
from src import config
from src.utils.cache import load_cache, save_cache
from src.utils.token_tracker import token_tracker

# ─── Logging & paths ─────────────────────────────────────────────────────────
setup_run_logging()
logger = logging.getLogger("run_on_off")

# --- ADJUSTED FILE PATHS ------------------------------------------------------
INPUT_FILE = Path('data/outputs/parsed_ingredients_output.json')
OUTPUT_FILE = Path('data/outputs/standardized_ingredients_output.json')
# ------------------------------------------------------------------------------

# Limit number of products to process (set to None to process all)
PRODUCT_LIMIT = 1000

# ──────────────────────────────────────────────────────────────────────────────
async def main() -> None:
    """Entry point for asynchronous execution."""
    logger.info("Starting ingredient standardisation …")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load cache: maps query ➔ {'id': str, 'label': str} or None
    cache: Dict[str, Optional[Dict[str, str]]] = load_cache(config.PIPELINE_CACHE_PATH)

    # Load input data
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}. Please create this file with your data.")
        return
    with INPUT_FILE.open('r', encoding='utf-8') as f:
        raw_product_data: Dict[str, Dict] = json.load(f)

    # Adapt and limit product data
    product_ingredients = {
        pid: data.get("parsed_ingredients", [])
        for pid, data in raw_product_data.items()
        if "parsed_ingredients" in data
    }
    logger.info(f"Loaded {len(product_ingredients)} products from input file.")

    if PRODUCT_LIMIT is not None:
        limited_items = list(product_ingredients.items())[:PRODUCT_LIMIT]
        product_ingredients = dict(limited_items)
        raw_product_data = {pid: raw_product_data[pid] for pid, _ in limited_items}
        logger.warning(f"Processing limited to {PRODUCT_LIMIT} products for this run.")

    # Initialise pipeline, semaphore, and the cache lock
    pipeline = create_pipeline(config.PIPELINE)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    cache_lock = asyncio.Lock()

    try:
        # Pass the lock to each concurrent task
        tasks = [process_product(pid, ings, pipeline, semaphore, cache, cache_lock)
                 for pid, ings in product_ingredients.items() if ings]
        
        results = await asyncio_tqdm.gather(*tasks, desc="Standardising Products")

        # Integrate results back into the original data structure
        for pid, standardized_list in results:
            if pid in raw_product_data and standardized_list is not None:
                raw_product_data[pid]['standardized_ingredients'] = standardized_list

        # Write the final output file
        with OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(raw_product_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Standardised data written to {OUTPUT_FILE}")

    finally:
        pipeline.close()
        logger.info(token_tracker.report_usage())
        logger.info("Run complete.")

# ─── Helpers ─────────────────────────────────────────────────────────────────
async def process_product(product_id: str,
                          ingredients: List[str],
                          pipeline,
                          semaphore: asyncio.Semaphore,
                          cache: Dict[str, Optional[Dict[str, str]]],
                          cache_lock: asyncio.Lock):
    """Return product_id and list of standardised mapping objects for its ingredients."""

    unique_ingredients = list(dict.fromkeys(ingredients))
    tasks: List[asyncio.Task] = []
    queries_for_tasks: List[str] = []

    for ing in unique_ingredients:
        if ing not in cache:
            logger.debug(f"Cache miss: scheduling pipeline run for '{ing}'")
            task = pipeline.run(query=ing, semaphore=semaphore)
            tasks.append(asyncio.create_task(task))
            queries_for_tasks.append(ing)
        else:
            logger.debug(f"Cache hit for: '{ing}'")

    if tasks:
        new_entries_added = False
        results = await asyncio.gather(*tasks)
        for query, res in zip(queries_for_tasks, results):
            mapping_result, _candidates = res if res else (None, None)
            # handle list of ranked results
            if isinstance(mapping_result, list):
                mapping_result = mapping_result[0] if mapping_result else None
            
            # --- CHANGED LOGIC ---
            # Only cache matches that are high-confidence.
            if mapping_result and mapping_result.get('confidence_score', 0.0) >= config.CONFIDENCE_THRESHOLD:
                # If high-confidence, create a small dict with the ID and Label to cache.
                cache_entry = {
                    'id': mapping_result.get('id'),
                    'label': mapping_result.get('label')
                }
                cache[query] = cache_entry
                new_entries_added = True  # Mark that a valid entry was added.
            
            # --- REMOVED ---
            # The 'else' block that stored 'None' in the cache has been removed.
            # Low-confidence or null results will no longer be saved to the cache file.

        # Save the updated cache to disk only if new high-confidence items were added.
        if new_entries_added:
            async with cache_lock:
                logger.debug(f"Product {product_id} saving updated cache with new high-confidence entries.")
                save_cache(config.PIPELINE_CACHE_PATH, cache)

    # Map every ingredient to its final mapping object (or None)
    # If a low-confidence ingredient was processed, it won't be in the cache,
    # so cache.get(ing) will correctly return None for the final output list.
    standardized_list = [cache.get(ing) for ing in ingredients]

    logger.info(f"Product {product_id}: {len(ingredients)} ingredients processed.")
    return product_id, standardized_list

# ─── Script execution guard ─────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
