# src/evaluation/evaluate_pipeline.py

import xml.etree.ElementTree as ET
import os
import sys
import logging
import json
from typing import Tuple, List, Dict

# Add project root to Python path to allow direct imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.retriever.hybrid_retriever import HybridRetriever
from src.rag_selectors.ollama_selector import OllamaSelector # Import the new selector
from src.rag_selectors.gemini_selector import GeminiSelector # Import the old selector for reference

from src.config import (
    ONTOLOGY_DUMP_JSON,
    WHOOSH_INDEX_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CURIE_PREFIX_MAP,
    DEFAULT_K_LEXICAL,
    DEFAULT_K_VECTOR,
    OLLAMA_SELECTOR_MODEL_NAME # Import the model name
)
from src.utils.ontology_utils import uri_to_curie

# --- Configuration for this specific evaluation script ---
EVALUATION_XML_FILE = os.path.join(PROJECT_ROOT, "data", "CafeteriaFCD_foodon_unique.xml")
# File to save detailed results of incorrect selections
EVALUATION_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "evaluation_results.json")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Re-used from evaluate_retriever_recall.py (no changes needed) ---
def parse_evaluation_xml(xml_file_path: str) -> list:
    """
    Parses the evaluation XML file to extract entities and their ground truth semantic tags.
    (This function is identical to the one in evaluate_retriever_recall.py)
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
                    true_curies = {uri_to_curie(uri, CURIE_PREFIX_MAP) for uri in true_uris} - {None}
                    
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

# --- New Evaluation Logic for the Full Pipeline ---
def evaluate_full_pipeline(
    retriever: HybridRetriever, 
    selector: GeminiSelector,  #
    gold_standard_data: list, 
    lexical_k: int, 
    vector_k: int
) -> Tuple[float, int, int, int, int, List[Dict]]:
    """
    Evaluates the full retrieval and selection pipeline against the gold standard data.

    Returns:
        A tuple containing:
        - accuracy_score (float)
        - total_entities_processed (int)
        - correct_selections (int)
        - retrieval_failures (int): Count of queries that returned no candidates.
        - selection_failures (int): Count of queries where the selector failed.
        - incorrect_selections (list): A list of dictionaries detailing the misses.
    """
    total_entities_processed = 0
    correct_selections = 0
    retrieval_failures = 0
    selection_failures = 0
    incorrect_selections = []

    if not gold_standard_data:
        logger.warning("No gold standard data provided for evaluation.")
        return 0.0, 0, 0, 0, 0, []

    for i, item in enumerate(gold_standard_data):
        query_text = item['text']
        true_curies = item['true_curies']
        
        total_entities_processed += 1
        logger.info(f"--- Processing ({i+1}/{len(gold_standard_data)}): '{query_text}' (True: {true_curies}) ---")

        # 1. RETRIEVAL STEP: Get candidates
        try:
            # Note: The HybridRetriever's search method should return a combined, reranked list of candidates.
            # We adapt to the provided retriever's output format.
            retriever_output_dict = retriever.search(
                query_string=query_text, lexical_limit=lexical_k, vector_k=vector_k
            )
            lexical_results = retriever_output_dict.get("lexical_results", [])
            vector_results = retriever_output_dict.get("vector_results", [])

            # Combine and deduplicate
            candidates = []
            seen_ids = set()
            for doc in lexical_results + vector_results:
                doc_id = doc.get('id')
                if doc_id and doc_id not in seen_ids:
                    candidates.append(doc)
                    seen_ids.add(doc_id)
            
            if not candidates:
                logger.warning(f"Retrieval Failure: No candidates found for '{query_text}'.")
                retrieval_failures += 1
                continue

        except Exception as e:
            logger.error(f"Error during retrieval for query '{query_text}': {e}", exc_info=True)
            retrieval_failures += 1
            continue

        # 2. SELECTION STEP: Use OllamaSelector
        try:
            selection_result = selector.select_best_term(query=query_text, candidates=candidates)

            if not selection_result or "chosen_id" not in selection_result:
                logger.warning(f"Selection Failure: Selector did not return a valid choice for '{query_text}'.")
                selection_failures += 1
                continue
            
            chosen_curie = selection_result["chosen_id"]

        except Exception as e:
            logger.error(f"Error during selection for query '{query_text}': {e}", exc_info=True)
            selection_failures += 1
            continue
        
        # 3. COMPARISON STEP
        if chosen_curie in true_curies:
            correct_selections += 1
            logger.info(f"✅ HIT! Query: '{query_text}'. Chosen: '{chosen_curie}'. Correct.")
        else:
            logger.info(f"❌ MISS! Query: '{query_text}'. Chosen: '{chosen_curie}', Expected: {true_curies}.")
            incorrect_selections.append({
                "query": query_text,
                "chosen_curie": chosen_curie,
                "true_curies": list(true_curies),
                "explanation": selection_result.get("explanation", "N/A"),
                "candidates_provided": [cand['id'] for cand in candidates]
            })

    if (total_entities_processed - retrieval_failures - selection_failures) == 0:
        accuracy_score = 0.0
    else:
        # Accuracy is based on the number of times the selector could make a choice
        accuracy_score = correct_selections / (total_entities_processed - retrieval_failures - selection_failures)

    return accuracy_score, total_entities_processed, correct_selections, retrieval_failures, selection_failures, incorrect_selections

def main():
    logger.info("Starting Full Pipeline Evaluation Script...")
    
    # 1. Check for necessary model name in config
    if not OLLAMA_SELECTOR_MODEL_NAME:
        logger.error("OLLAMA_SELECTOR_MODEL_NAME is not set in config.py. Exiting.")
        return

    # 2. Parse Gold Standard XML
    logger.info(f"Loading gold standard data from: {EVALUATION_XML_FILE}")
    gold_standard_data = parse_evaluation_xml(EVALUATION_XML_FILE)
    if not gold_standard_data:
        logger.error("Failed to load or parse gold standard data. Exiting.")
        return

    # 3. Initialize Pipeline Components
    try:
        logger.info("Initializing HybridRetriever...")
        retriever = HybridRetriever(
            ontology_data_path=ONTOLOGY_DUMP_JSON,
            whoosh_index_dir=WHOOSH_INDEX_DIR,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_metadata_path=FAISS_METADATA_PATH,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )
        logger.info("HybridRetriever initialized successfully.")

        logger.info(f"Initializing OllamaSelector with model '{OLLAMA_SELECTOR_MODEL_NAME}'...")
        selector = GeminiSelector(retriever=retriever)
        logger.info("OllamaSelector initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline components: {e}", exc_info=True)
        return

    # 4. Perform Evaluation
    logger.info(
        f"Starting evaluation with Retriever(lexical_k={DEFAULT_K_LEXICAL}, vector_k={DEFAULT_K_VECTOR}) "
        f"and Selector(model={OLLAMA_SELECTOR_MODEL_NAME})"
    )
    
    accuracy, total, correct, ret_fails, sel_fails, misses = evaluate_full_pipeline(
        retriever, 
        selector, 
        gold_standard_data, 
        lexical_k=DEFAULT_K_LEXICAL,
        vector_k=DEFAULT_K_VECTOR
    )

    # 5. Print and Save Results
    logger.info("--- Evaluation Complete ---")
    logger.info(f"Total entities evaluated: {total}")
    logger.info(f"Retrieval Failures (no candidates): {ret_fails}")
    logger.info(f"Selection Failures (LLM error): {sel_fails}")
    logger.info("-" * 27)
    valid_attempts = total - ret_fails - sel_fails
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

if __name__ == "__main__":
    if not os.path.exists(EVALUATION_XML_FILE):
        logger.error(f"Evaluation XML file '{EVALUATION_XML_FILE}' not found.")
    else:
        main()