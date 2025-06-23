# scripts/map_ingredients.py
"""
This script loads extracted ingredient entities from a JSON file,
runs them through the RAG pipeline to link them to ontology terms,
and saves the structured results to a new JSON file.
"""

import os
import sys
import json
import logging
from tqdm import tqdm

# --- Add project root to sys.path ---
# This allows the script to import modules from the 'src' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.gemini_pipeline import RAGPipeline
from src import config

# --- Configuration ---
LOGGING_LEVEL = logging.INFO
# Limit the number of products to process. Set to None to process all.
PRODUCT_LIMIT = 5 

INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'outputs', 'parsed_ingredients_output.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'outputs', 'mapped_ingredients_output.json')


# --- Setup Logging ---
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the ingredient mapping process.
    """
    logger.info("Starting ingredient to ontology mapping process...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    pipeline = None
    try:
        # --- 1. Load Input Data ---
        logger.info(f"Loading ingredients from: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Input file not found: {INPUT_FILE}")
            logger.error("Please ensure you have run the parsing script first.")
            return

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_ingredients_data = json.load(f)

        # --- 2. Initialize RAG Pipeline ---
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully.")

        # --- 3. Process Ingredients ---
        all_mappings = {}
        
        # Get items to process, applying the limit if specified
        items_to_process = list(all_ingredients_data.items())
        if PRODUCT_LIMIT is not None:
            items_to_process = items_to_process[:PRODUCT_LIMIT]
            logger.warning(f"Processing a limited set of {PRODUCT_LIMIT} products.")

        # Use tqdm for a progress bar
        for product_id, ingredients in tqdm(items_to_process, desc="Processing Products"):
            logger.info(f"--- Processing Product ID: {product_id} ---")
            
            product_mappings = []
            unique_ingredients = sorted(list(set(ingredients))) # Process unique ingredients to avoid duplicate work

            for ingredient_query in unique_ingredients:
                logger.info(f"Querying for: '{ingredient_query}'")
                
                # Run the pipeline for the ingredient string
                mapping_result = pipeline.run(query=ingredient_query)
                
                if mapping_result:
                    logger.info(f"  -> Found mapping: '{mapping_result.get('label')}' (ID: {mapping_result.get('id')})")
                else:
                    logger.warning(f"  -> No mapping found for '{ingredient_query}'")
                
                # Store the result in a structured format
                product_mappings.append({
                    "original_ingredient": ingredient_query,
                    "mapping_result": mapping_result if mapping_result else "No mapping found"
                })
            
            all_mappings[product_id] = product_mappings

        # --- 4. Save Results ---
        logger.info(f"Saving mapped results to: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
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