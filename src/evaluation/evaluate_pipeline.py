# src/evaluation/evaluate_pipeline.py

import xml.etree.ElementTree as ET
import os
import sys
import logging
import json
from typing import Tuple, List, Dict

# Add project root to Python path to allow direct imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- MODIFIED IMPORTS ---
# Import the pipeline factory and base class instead of individual components
from src.pipeline.pipeline_factory import get_pipeline
from src.pipeline.base_pipeline import BaseRAGPipeline
from src import config
from src.utils.ontology_utils import uri_to_curie

# --- Configuration for this specific evaluation script ---
EVALUATION_XML_FILE = os.path.join(PROJECT_ROOT, "data", "CafeteriaFCD_foodon_unique.xml")
# File to save detailed results of incorrect selections
EVALUATION_OUTPUT_FILE = os.path.join(PROJECT_ROOT, f"evaluation_results_{config.PIPELINE}.json")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Re-used from evaluate_retriever_recall.py (no changes needed) ---
def parse_evaluation_xml(xml_file_path: str) -> list:
    """
    Parses the evaluation XML file to extract entities and their ground truth semantic tags.
    """
    if not os.path.exists(xml_file_path):
        logger.error(f"Evaluation XML file not found: {xml_file_path}")
        return []

    gold_standard_data = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for doc_idx, document_node in enumerate(root.findall('.//document')):
            doc_id_node = document_node.find('id')
            doc_id = doc_id_node.text if doc_id_node is not None else f"doc_{doc_idx}"

            for ann_idx, annotation_node in enumerate(document_node.findall('annotation')):
                entity_text_node = annotation_node.find('text')
                semantic_tags_node = annotation_node.find('infon[@key="semantic_tags"]')
                ann_id_val = annotation_node.get('id', f"ann_{doc_idx}_{ann_idx}")

                if entity_text_node is not None and semantic_tags_node is not None and entity_text_node.text is not None and semantic_tags_node.text is not None:
                    entity_text = entity_text_node.text.strip()
                    raw_tags = semantic_tags_node.text.strip()
                    true_uris = {tag.strip() for tag in raw_tags.split(';') if tag.strip()}
                    true_curies = {uri_to_curie(uri, config.CURIE_PREFIX_MAP) for uri in true_uris} - {None}

                    if entity_text and true_curies:
                        gold_standard_data.append({
                            'text': entity_text,
                            'true_curies': true_curies,
                            'doc_id': doc_id,
                            'ann_id': ann_id_val
                        })
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []

    logger.info(f"Successfully parsed {len(gold_standard_data)} entities from {xml_file_path}")
    return gold_standard_data

# --- REFACTORED EVALUATION LOGIC ---
def evaluate_full_pipeline(
    pipeline: BaseRAGPipeline,
    gold_standard_data: list
) -> Tuple[float, int, int, int, List[Dict]]:
    """
    Evaluates the full RAG pipeline against the gold standard data using the pipeline's run method.

    Returns:
        A tuple containing:
        - accuracy (float): The final accuracy score.
        - total_entities (int): Total items processed.
        - hits (int): Number of correct selections.
        - failures (int): Number of times the pipeline failed to return a result.
        - misses (list): A list of dictionaries detailing the incorrect selections.
    """
    total_entities = len(gold_standard_data)
    if total_entities == 0:
        logger.warning("No gold standard data provided for evaluation.")
        return 0.0, 0, 0, 0, []

    hits = 0
    failures = 0
    misses = []

    for i, item in enumerate(gold_standard_data):
        query = item['text']
        true_curies = item['true_curies']
        logger.info(f"--- Processing ({i+1}/{total_entities}): '{query}' ---")

        try:
            # Use the pipeline's run method directly to get the final result
            result_tuple = pipeline.run(query=query, rerank_top_n=50)

            if not result_tuple or not result_tuple[0]:
                logger.warning(f"Failure: Pipeline returned no result for '{query}'.")
                failures += 1
                continue

            final_result, candidates = result_tuple
            chosen_curie = final_result.get('id')

            if chosen_curie in true_curies:
                hits += 1
                logger.info(f"✅ HIT! Query: '{query}'. Chosen: '{chosen_curie}'.")
            else:
                logger.info(f"❌ MISS! Query: '{query}'. Chosen: '{chosen_curie}', Expected: {true_curies}.")
                misses.append({
                    "query": query,
                    "chosen_curie": chosen_curie,
                    "true_curies": list(true_curies),
                    "explanation": final_result.get("explanation", "N/A"),
                    "confidence_score": final_result.get("confidence_score", 0.0),
                    "candidates_provided": [c.get('id') for c in candidates if c.get('id')]
                })

        except Exception as e:
            logger.error(f"An unhandled error occurred while processing query '{query}': {e}", exc_info=True)
            failures += 1
            continue

    valid_attempts = total_entities - failures
    accuracy = hits / valid_attempts if valid_attempts > 0 else 0.0
    return accuracy, total_entities, hits, failures, misses

# --- REFACTORED MAIN FUNCTION ---
def main():
    logger.info("Starting Full Pipeline Evaluation Script...")
    logger.info(f"Evaluating Pipeline: '{config.PIPELINE}'")

    # 1. Parse Gold Standard XML
    logger.info(f"Loading gold standard data from: {EVALUATION_XML_FILE}")
    gold_standard_data = parse_evaluation_xml(EVALUATION_XML_FILE)
    if not gold_standard_data:
        logger.error("Failed to load or parse gold standard data. Exiting.")
        return

    pipeline = None
    try:
        # 2. Initialize Pipeline using the factory
        logger.info(f"Initializing RAG pipeline: '{config.PIPELINE}'...")
        pipeline = get_pipeline(config.PIPELINE)
        logger.info("Pipeline initialized successfully.")

        # 3. Perform Evaluation
        logger.info(f"Starting evaluation for {len(gold_standard_data)} entities...")
        accuracy, total, correct, failures, misses = evaluate_full_pipeline(
            pipeline,
            gold_standard_data
        )

        # 4. Print and Save Results
        logger.info("--- Evaluation Complete ---")
        logger.info(f"Total entities evaluated: {total}")
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
        # 5. Clean up pipeline resources
        if pipeline:
            logger.info("Closing pipeline resources...")
            pipeline.close()

if __name__ == "__main__":
    if not os.path.exists(EVALUATION_XML_FILE):
        logger.error(f"Evaluation XML file '{EVALUATION_XML_FILE}' not found.")
    else:
        main()