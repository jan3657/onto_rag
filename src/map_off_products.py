# ------------------------------------------------------------------------------
#!/usr/bin/env python
"""
run_on_off.py – Map parsed ingredient lists to standardized ontology labels using OntoRag.

*** MODIFIED TO WORK WITH CUSTOM DATA STRUCTURE AND RETURN None FOR NO MATCH ***

Input  : your_data.json (product_id ➔ { "parsed_ingredients": list[str], ... })
Output : your_data_standardized.json (product_id ➔ { ..., "standardized_ingredients": list[str|None] })

The script re‑uses the OntoRag RAG pipeline, leverages a persistent cache to avoid
re‑processing identical ingredient queries, and processes products concurrently with
an asyncio‑compatible semaphore governed by config.MAX_CONCURRENT_REQUESTS.

The script preserves the original input data and adds the standardized labels
under a new 'standardized_ingredients' key. If an ingredient is not found,
it is mapped to None.
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.asyncio import tqdm as asyncio_tqdm

# ─── Add project root to sys.path ─────────────────────────────────────────────
# Ensure this points to your project's root directory where 'src' is located.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── OntoRag imports ─────────────────────────────────────────────────────────
# These imports depend on your project structure. Make sure they are correct.
from src.pipeline.pipeline_factory import get_pipeline  # type: ignore
from src import config  # type: ignore
from src.utils.cache import load_cache, save_cache  # type: ignore

# ─── Logging & paths ─────────────────────────────────────────────────────────
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("run_on_off")

# --- ADJUSTED FILE PATHS ------------------------------------------------------
# Place this script in a directory, and put your data file in the same directory.
INPUT_FILE = Path('data/outputs/parsed_ingredients_output.json')
OUTPUT_FILE = Path('data/outputs/standardized_ingredients_output_5_samples.json')
# ------------------------------------------------------------------------------

# Limit number of products to process (set to None to process all)
PRODUCT_LIMIT = 10  # limit to 10 products for quick tests

# ──────────────────────────────────────────────────────────────────────────────
async def main() -> None:
    """Entry point for asynchronous execution."""
    logger.info("Starting ingredient standardisation …")

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load cache
    cache: Dict[str, Optional[str]] = load_cache(config.PIPELINE_CACHE_PATH)  # maps query ➔ label or None

    # Load input data
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}. Please create this file with your data.")
        return
    with INPUT_FILE.open('r', encoding='utf-8') as f:
        # Load the raw data which has a nested structure
        raw_product_data: Dict[str, Dict] = json.load(f)

    # Adapt the raw data to extract the 'parsed_ingredients' list
    product_ingredients = {
        pid: data.get("parsed_ingredients", [])
        for pid, data in raw_product_data.items()
        if "parsed_ingredients" in data
    }
    logger.info(f"Loaded {len(product_ingredients)} products from input file.")

    # Apply product limit if specified
    if PRODUCT_LIMIT is not None:
        limited_items = list(product_ingredients.items())[:PRODUCT_LIMIT]
        product_ingredients = dict(limited_items)
        # We also need to filter the raw_product_data for the final output
        raw_product_data = {pid: raw_product_data[pid] for pid, _ in limited_items}
        logger.warning(f"Processing limited to {PRODUCT_LIMIT} products for this run.")

    # Initialise pipeline
    pipeline = get_pipeline(config.PIPELINE)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    try:
        tasks = [process_product(pid, ings, pipeline, semaphore, cache)
                 for pid, ings in product_ingredients.items() if ings]
        
        results = await asyncio_tqdm.gather(*tasks, desc="Standardising Products")

        # Integrate results back into the original data structure
        for pid, standardized_list in results:
            if pid in raw_product_data and standardized_list is not None:
                raw_product_data[pid]['standardized_ingredients'] = standardized_list

        # Write the modified original data structure to the output file
        with OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(raw_product_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Standardised data written to {OUTPUT_FILE}")

    finally:
        pipeline.close()
        save_cache(config.PIPELINE_CACHE_PATH, cache)
        logger.info("Run complete.")

# ─── Helpers ─────────────────────────────────────────────────────────────────
async def process_product(product_id: str,
                          ingredients: List[str],
                          pipeline,
                          semaphore: asyncio.Semaphore,
                          cache: Dict[str, Optional[str]]):
    """Return product_id and list of standardised labels for its ingredients."""

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
        results = await asyncio.gather(*tasks)
        for query, res in zip(queries_for_tasks, results):
            mapping_result, _candidates = res if res else (None, None)
            label = (mapping_result or {}).get('label') if mapping_result else None
            
            # *** THIS IS THE KEY CHANGE ***
            # Store the label if found, otherwise store None.
            cache[query] = label

    # Map every ingredient to its final label (or None)
    standardized_list = [cache.get(ing) for ing in ingredients]

    logger.info(f"Product {product_id}: {len(ingredients)} ingredients processed.")
    return product_id, standardized_list

# ─── Script execution guard ─────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())