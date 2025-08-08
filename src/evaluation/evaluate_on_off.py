# scripts/run_on_off.py
"""
Script to run the RAG-based ontology mapping pipeline on a batch of products
from an input file (e.g., parsed Open Food Facts data).

It processes each product's ingredients, links them to ontology terms, and
saves the detailed structured results, including candidates considered by the LLM,
to an output JSON file. It now uses a persistent cache to avoid re-processing ingredients.
"""

import sys
import json
import logging
import asyncio
from tqdm.asyncio import tqdm as asyncio_tqdm
from pathlib import Path


# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.pipeline.pipeline_factory import get_pipeline
from src import config
# --- NEW: Import cache utilities ---
from src.utils.cache import load_cache, save_cache
# -----------------------------------

# --- Configuration ---
LOGGING_LEVEL = logging.INFO
PRODUCT_LIMIT = 2 # Increased for a better demo

# --- Paths Configuration (using pathlib) ---
DATA_DIR = PROJECT_ROOT / 'data' / 'outputs'
INPUT_FILE = DATA_DIR / 'parsed_ingredients_output.json'
OUTPUT_FILE = DATA_DIR / 'mapped_ingredients_output_2_samples.json'


# --- Setup Logging ---
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# --- HELPER FUNCTION (Unchanged) ---
def simplify_mapping_result(result):
    """
    Removes verbose keys from a mapping result dictionary to keep the output clean.
    Returns the simplified dictionary or the original input if it's not a dict.
    """
    if not isinstance(result, dict):
        return result
    keys_to_remove = {'ancestors', 'parents', 'relations'}
    simplified_dict = {
        key: value for key, value in result.items()
        if key not in keys_to_remove
    }
    return simplified_dict


async def main():
    """
    Main function to run the batch ingredient mapping process.
    """
    logger.info("Starting BATCH ontology mapping process...")
    logger.info(f"Using pipeline: {config.PIPELINE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pipeline = None
    # --- NEW: Load the cache at the start ---
    cache = load_cache(config.PIPELINE_CACHE_PATH)
    # -----------------------------------------
    try:
        # --- 1. Load Input Data ---
        logger.info(f"Loading ingredients from: {INPUT_FILE}")
        if not INPUT_FILE.exists():
            logger.error(f"Input file not found: {INPUT_FILE}")
            logger.error("Please ensure you have run the parsing script first.")
            return

        with INPUT_FILE.open('r', encoding='utf-8') as f:
            all_product_data = json.load(f)

        # --- 2. Initialize RAG Pipeline ---
        logger.info("Initializing RAG pipeline...")
        pipeline = get_pipeline(config.PIPELINE)
        logger.info("RAG pipeline initialized successfully.")

        # --- 3. Process Ingredients Concurrently ---
        all_mappings = {}
        
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        items_to_process = list(all_product_data.items())
        if PRODUCT_LIMIT is not None:
            items_to_process = items_to_process[:PRODUCT_LIMIT]
            logger.warning(f"Processing a limited set of {PRODUCT_LIMIT} products.")

        product_tasks = []
        for product_id, product_data in items_to_process:
            # --- NEW: Pass the cache to the processing function ---
            product_tasks.append(
                process_single_product(pipeline, product_id, product_data, semaphore, cache)
            )
            # ----------------------------------------------------

        results = await asyncio_tqdm.gather(*product_tasks, desc="Processing Products")

        for product_id, product_result in results:
            if product_result:
                all_mappings[product_id] = product_result

        # --- 4. Save Results ---
        logger.info(f"Saving mapped results to: {OUTPUT_FILE}")
        with OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(all_mappings, f, indent=4)

        logger.info("Mapping process completed successfully!")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- 5. Clean up and Save Cache ---
        if pipeline:
            logger.info("Closing pipeline resources.")
            pipeline.close()
        # --- NEW: Save the updated cache at the end ---
        save_cache(config.PIPELINE_CACHE_PATH, cache)
        # --------------------------------------------

async def process_single_product(pipeline, product_id, product_data, semaphore, cache):
    """
    Asynchronously processes all ingredients for a single product, using a cache.
    """
    original_ingredients_text = product_data.get("original_ingredients", "")
    parsed_ingredients = product_data.get("parsed_ingredients", [])
    unique_ingredients = sorted(list(set(parsed_ingredients)))

    # --- NEW CACHING LOGIC ---
    tasks_to_run = []
    queries_for_tasks = []
    # This dictionary will hold all results, whether from cache or a new run.
    all_ingredient_results = {}

    # 1. First, check the cache for each ingredient
    for ingredient_query in unique_ingredients:
        if ingredient_query in cache:
            logger.info(f"CACHE HIT for query: '{ingredient_query}' in Product {product_id}")
            all_ingredient_results[ingredient_query] = cache[ingredient_query]
        else:
            logger.info(f"CACHE MISS for query: '{ingredient_query}' in Product {product_id}. Scheduling pipeline run.")
            # This query needs to be run by the pipeline.
            task = pipeline.run(query=ingredient_query, semaphore=semaphore)
            tasks_to_run.append(task)
            queries_for_tasks.append(ingredient_query)

    # 2. Run the pipeline concurrently for all cache misses
    if tasks_to_run:
        logger.info(f"Executing pipeline for {len(tasks_to_run)} cache misses in Product {product_id}.")
        pipeline_results = await asyncio.gather(*tasks_to_run)
        
        # 3. Process new results and update both the local results and the main cache
        for i, result_tuple in enumerate(pipeline_results):
            query = queries_for_tasks[i]
            all_ingredient_results[query] = result_tuple
            # Only cache successful runs to avoid populating with failures
            if result_tuple and result_tuple[0]:
                cache[query] = result_tuple

    # 4. Build the final output for the product using all results (cached and new)
    mapped_ingredients = []
    for ingredient_query in unique_ingredients: # Iterate in original order
        result_tuple = all_ingredient_results.get(ingredient_query)
        mapping_result, candidates = (None, []) if not result_tuple else result_tuple
        simplified_mapping = simplify_mapping_result(mapping_result)

        if mapping_result:
            logger.info(f"Product {product_id}: Query '{ingredient_query}' -> Mapped to '{simplified_mapping.get('label')}'")
        else:
            logger.warning(f"Product {product_id}: Query '{ingredient_query}' -> No mapping found")
            
        mapped_ingredients.append({
            "original_ingredient": ingredient_query,
            "mapping_result": simplified_mapping if simplified_mapping else "No mapping found",
            "candidates": candidates if candidates else []
        })

    product_output = {
        "original_ingredients": original_ingredients_text,
        "mapped_ingredients": mapped_ingredients
    }
    return product_id, product_output


if __name__ == "__main__":
    asyncio.run(main())