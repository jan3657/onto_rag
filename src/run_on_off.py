# scripts/run_on_off.py
"""
Script to run the RAG-based ontology mapping pipeline on a batch of products
from an input file (e.g., parsed Open Food Facts data).

It processes each product's ingredients, links them to ontology terms, and
saves the detailed structured results, including candidates considered by the LLM,
to an output JSON file. The final mapping result is simplified to improve readability.
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
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        items_to_process = list(all_product_data.items())
        if PRODUCT_LIMIT is not None:
            items_to_process = items_to_process[:PRODUCT_LIMIT]
            logger.warning(f"Processing a limited set of {PRODUCT_LIMIT} products.")

        # Create a list of tasks for all products
        product_tasks = []
        for product_id, product_data in items_to_process:
            product_tasks.append(
                process_single_product(pipeline, product_id, product_data, semaphore)
            )

        # Execute all product processing tasks concurrently with a progress bar
        results = await asyncio_tqdm.gather(*product_tasks, desc="Processing Products")

        # Collate results
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
        # --- 5. Clean up ---
        if pipeline:
            logger.info("Closing pipeline resources.")
            pipeline.close()

async def process_single_product(pipeline, product_id, product_data, semaphore):
    """
    Asynchronously processes all ingredients for a single product.
    """
    original_ingredients_text = product_data.get("original_ingredients", "")
    parsed_ingredients = product_data.get("parsed_ingredients", [])
    unique_ingredients = sorted(list(set(parsed_ingredients)))

    ingredient_tasks = []
    for ingredient_query in unique_ingredients:
        # Create a task for each ingredient, passing the semaphore
        task = pipeline.run(query=ingredient_query, semaphore=semaphore)
        ingredient_tasks.append(asyncio.create_task(task))

    # Wait for all ingredients of this product to be mapped
    results = await asyncio.gather(*ingredient_tasks)

    mapped_ingredients = []
    for i, ingredient_query in enumerate(unique_ingredients):
        result_tuple = results[i]
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