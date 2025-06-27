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
from tqdm import tqdm
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# --- Configuration ---
LOGGING_LEVEL = logging.INFO
PRODUCT_LIMIT = 100

# --- Paths Configuration (using pathlib) ---
DATA_DIR = PROJECT_ROOT / 'data' / 'outputs'
INPUT_FILE = DATA_DIR / 'parsed_ingredients_output.json'
OUTPUT_FILE = DATA_DIR / 'mapped_ingredients_output.json'


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


def main():
    """
    Main function to run the batch ingredient mapping process.
    """
    logger.info("Starting batch ontology mapping process...")
    logger.info(f"Using pipeline: {config.PIPELINE.__name__}")

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
            ## MODIFIED ##: Renamed variable to reflect new data structure
            all_product_data = json.load(f)

        # --- 2. Initialize RAG Pipeline ---
        logger.info("Initializing RAG pipeline...")
        pipeline = config.PIPELINE()
        logger.info("RAG pipeline initialized successfully.")

        # --- 3. Process Ingredients ---
        all_mappings = {}

        ## MODIFIED ##: Use the new data variable
        items_to_process = list(all_product_data.items())
        if PRODUCT_LIMIT is not None:
            items_to_process = items_to_process[:PRODUCT_LIMIT]
            logger.warning(f"Processing a limited set of {PRODUCT_LIMIT} products.")

        ## MODIFIED ##: Loop now unpacks the product data dictionary
        for product_id, product_data in tqdm(items_to_process, desc="Processing Products"):
            logger.debug(f"--- Processing Product ID: {product_id} ---")

            ## ADDED ##: Extract original and parsed ingredients from the product data
            original_ingredients_text = product_data.get("original_ingredients", "")
            parsed_ingredients = product_data.get("parsed_ingredients", [])

            mapped_ingredients = [] # Will store the list of mapping results for this product
            
            # Continue to process only unique ingredients to avoid redundant work
            unique_ingredients = sorted(list(set(parsed_ingredients)))

            for ingredient_query in unique_ingredients:
                logger.debug(f"Querying for: '{ingredient_query}'")

                result_tuple = pipeline.run(query=ingredient_query)
                mapping_result, candidates = (None, []) if not result_tuple else result_tuple

                simplified_mapping = simplify_mapping_result(mapping_result)

                if mapping_result:
                    logger.info(f"Query: '{ingredient_query}' -> Mapped to: '{mapping_result.get('label')}' (ID: {mapping_result.get('id')})")
                else:
                    logger.warning(f"Query: '{ingredient_query}' -> No mapping found")

                # Store the simplified result and candidates in the final JSON structure
                mapped_ingredients.append({
                    "original_ingredient": ingredient_query,
                    "mapping_result": simplified_mapping if simplified_mapping else "No mapping found",
                    "candidates": candidates if candidates else []
                })

            ## MODIFIED ##: Structure the output to include original ingredients text
            all_mappings[product_id] = {
                "original_ingredients": original_ingredients_text,
                "mapped_ingredients": mapped_ingredients
            }

        # --- 4. Save Results ---
        logger.info(f"Saving mapped results to: {OUTPUT_FILE}")
        with OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(all_mappings, f, indent=4)

        logger.info("Mapping process completed successfully!")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the mapping process: {e}", exc_info=True)
    finally:
        # --- 5. Clean up ---
        if pipeline:
            logger.info("Closing pipeline resources.")
            pipeline.close()

if __name__ == "__main__":
    main()