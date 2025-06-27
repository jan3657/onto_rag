# src/evaluation/evaluate_retriever_recall.py

import xml.etree.ElementTree as ET
import os
import sys
import logging
from collections import defaultdict

# Add project root to Python path to allow direct imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.retriever.hybrid_retriever import HybridRetriever
from src.config import (
    ONTOLOGY_DUMP_JSON,
    WHOOSH_INDEX_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CURIE_PREFIX_MAP,
    DEFAULT_K_LEXICAL,
    DEFAULT_K_VECTOR,
    DEFAULT_RERANK_K
)
from src.utils.ontology_utils import uri_to_curie

# --- Configuration for this specific evaluation script ---
# Path to your evaluation XML file
EVALUATION_XML_FILE = os.path.join(PROJECT_ROOT, "data", "CafeteriaFCD_foodon_unique.xml")
# Top K results from the combined retriever output to consider for a match
RECALL_AT_K = DEFAULT_RERANK_K # How many retrieved items to check for a match

# --- Logging Setup ---
# Stays with basicConfig as src.utils.logging.get_logger is "to be developed"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_evaluation_xml(xml_file_path: str) -> list:
    """
    Parses the evaluation XML file to extract entities and their ground truth semantic tags.

    Args:
        xml_file_path (str): Path to the XML file.

    Returns:
        list: A list of dictionaries, each containing:
              {'text': str, 'true_curies': set_of_str, 'doc_id': str, 'ann_id': str}
              Returns an empty list if parsing fails.
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
            
            annotations = document_node.findall('annotation')
            for ann_idx, annotation_node in enumerate(annotations):
                entity_text_node = annotation_node.find('text')
                semantic_tags_node = annotation_node.find('infon[@key="semantic_tags"]')
                ann_id_val = annotation_node.get('id', f"ann_{doc_idx}_{ann_idx}")


                if entity_text_node is not None and semantic_tags_node is not None and entity_text_node.text is not None and semantic_tags_node.text is not None:
                    entity_text = entity_text_node.text.strip()
                    
                    raw_tags = semantic_tags_node.text.strip()
                    true_uris = {tag.strip() for tag in raw_tags.split(';') if tag.strip()}
                    
                    true_curies = set()
                    for uri in true_uris:
                        try:
                            # Adjusted: Use CURIE_PREFIX_MAP
                            curie = uri_to_curie(uri, CURIE_PREFIX_MAP)
                            if curie: 
                                true_curies.add(curie)
                            else:
                                logger.warning(f"Could not convert URI to CURIE: {uri} for entity '{entity_text}' in {doc_id} (ann: {ann_id_val})")
                        except Exception as e:
                            logger.error(f"Error converting URI {uri} to CURIE: {e}")
                    
                    if entity_text and true_curies:
                        gold_standard_data.append({
                            'text': entity_text,
                            'true_curies': true_curies,
                            'doc_id': doc_id,
                            'ann_id': ann_id_val
                        })
                    elif entity_text: # Has text but no convertible true CURIEs
                        logger.warning(f"Entity '{entity_text}' in {doc_id} (ann: {ann_id_val}) had no convertible true CURIEs from URIs: {true_uris}")

                # else: # This can be too verbose if many annotations don't have these specific fields
                #     logger.debug(f"Annotation missing text or semantic_tags in {doc_id}, ann_id={ann_id_val}")


    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during XML parsing: {e}", exc_info=True)
        return []
        
    logger.info(f"Successfully parsed {len(gold_standard_data)} entities with text and true CURIEs from {xml_file_path}")
    return gold_standard_data

def evaluate_retriever(retriever: HybridRetriever, gold_standard_data: list, recall_at_k: int, lexical_k: int, vector_k: int) -> tuple[float, int, int]:
    """
    Evaluates the retriever against the gold standard data.

    Args:
        retriever (HybridRetriever): The initialized hybrid retriever.
        gold_standard_data (list): List of gold standard entities and their CURIEs.
        recall_at_k (int): The K value for Recall@K (slice of combined results).
        lexical_k (int): Number of results to fetch from lexical search.
        vector_k (int): Number of results to fetch from vector search.

    Returns:
        tuple: (recall_score, total_entities_processed, hits)
    """
    total_entities_processed = 0
    hits = 0
    
    if not gold_standard_data:
        logger.warning("No gold standard data provided for evaluation.")
        return 0.0, 0, 0

    for i, item in enumerate(gold_standard_data):
        query_text = item['text']
        true_curies = item['true_curies']
        
        if not query_text or not true_curies:
            # This should ideally be filtered by parse_evaluation_xml already
            logger.warning(f"Skipping item with empty query text or true_curies: {item}")
            continue
            
        total_entities_processed += 1
        
        logger.debug(f"({i+1}/{len(gold_standard_data)}) Querying for: '{query_text}', True CURIEs: {true_curies}")

        try:
            # Adjusted: HybridRetriever.search returns a dict: {"lexical_results": [], "vector_results": []}
            retriever_output_dict = retriever.search(
                query_string=query_text,
                lexical_limit=lexical_k,
                vector_k=vector_k,
                target_ontologies=["foodon", "chebi"], # Adjusted: Use target_ontologies to limit search
            )
            
            lexical_results = retriever_output_dict.get("lexical_results", [])
            vector_results = retriever_output_dict.get("vector_results", [])

            # Combine and deduplicate results, lexical first then vector.
            # Scores are not comparable, so this is a simple merge strategy.
            # The 'id' field from result dicts contains the CURIE.
            combined_ordered_results = []
            seen_ids = set()

            for doc in lexical_results:
                doc_id = doc.get('id') # 'id' is the CURIE
                if doc_id and doc_id not in seen_ids:
                    combined_ordered_results.append(doc) # doc contains 'id', 'label', 'score', etc.
                    seen_ids.add(doc_id)
            
            for doc in vector_results:
                doc_id = doc.get('id') # 'id' is the CURIE
                if doc_id and doc_id not in seen_ids:
                    combined_ordered_results.append(doc)
                    seen_ids.add(doc_id)
            
            # Extract the CURIEs from the top `recall_at_k` combined documents
            # Adjusted: use doc['id'] as it stores the CURIE
            retrieved_curies_set = {doc['id'] for doc in combined_ordered_results[:recall_at_k]}
            logger.debug(f"Retrieved CURIEs (top {recall_at_k} from combined): {retrieved_curies_set}")

            if not true_curies.isdisjoint(retrieved_curies_set):
                hits += 1
                logger.info(f"HIT! Query: '{query_text}'. True: {true_curies}. Found in top {recall_at_k}: {true_curies.intersection(retrieved_curies_set)}")
            else:
                logger.info(f"MISS. Query: '{query_text}'. True: {true_curies}. Top {recall_at_k} (combined) CURIEs: {retrieved_curies_set}")
                # For misses, log more details if needed, e.g. full combined_ordered_results
                # logger.debug(f"Full combined/ordered results for miss: {combined_ordered_results}")


        except Exception as e:
            logger.error(f"Error during retrieval or processing for query '{query_text}': {e}", exc_info=True)
            
    if total_entities_processed == 0:
        logger.warning("No valid entities were processed for evaluation.")
        return 0.0, 0, 0
        
    recall_score = hits / total_entities_processed
    return recall_score, total_entities_processed, hits

def main():
    logger.info("Starting Retriever Evaluation Script...")

    # 1. Parse the Gold Standard XML
    logger.info(f"Loading gold standard data from: {EVALUATION_XML_FILE}")
    gold_standard_data = parse_evaluation_xml(EVALUATION_XML_FILE)
    if not gold_standard_data:
        logger.error("Failed to load or parse gold standard data. Exiting.")
        return

    # 2. Initialize the HybridRetriever
    logger.info("Initializing HybridRetriever...")
    try:
        # Ensure all paths are valid and files exist where expected by HybridRetriever
        # Adjusted: Use ONTOLOGY_DUMP_JSON for ontology_data_path
        if not os.path.exists(ONTOLOGY_DUMP_JSON):
            logger.error(f"Ontology dump not found: {ONTOLOGY_DUMP_JSON}. Run ingestion pipeline.")
            return
        if not os.path.exists(WHOOSH_INDEX_DIR) or not os.listdir(WHOOSH_INDEX_DIR): # Check if dir exists and is not empty
             logger.error(f"Whoosh index directory is empty or not found: {WHOOSH_INDEX_DIR}. Run ingestion pipeline.")
             return
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.error(f"FAISS index not found: {FAISS_INDEX_PATH}. Run ingestion pipeline.")
            return
        if not os.path.exists(FAISS_METADATA_PATH):
            logger.error(f"FAISS metadata not found: {FAISS_METADATA_PATH}. Run ingestion pipeline.")
            return

        retriever = HybridRetriever()
        logger.info("HybridRetriever initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize HybridRetriever: {e}", exc_info=True)
        return

    # 3. Perform Evaluation
    logger.info(f"Starting evaluation with Recall@{RECALL_AT_K}...")
    # Adjusted: Use DEFAULT_K_LEXICAL and DEFAULT_K_VECTOR
    logger.info(f"HybridRetriever search params: Lexical K={DEFAULT_K_LEXICAL}, Vector K={DEFAULT_K_VECTOR}")
    
    recall_score, total_entities, hits = evaluate_retriever(
        retriever, 
        gold_standard_data, 
        recall_at_k=RECALL_AT_K,
        lexical_k=DEFAULT_K_LEXICAL, # Adjusted
        vector_k=DEFAULT_K_VECTOR    # Adjusted
    )

    # 4. Print Results
    logger.info("--- Evaluation Results ---")
    logger.info(f"Total entities processed: {total_entities}")
    logger.info(f"Number of hits (at least one true CURIE found in top {RECALL_AT_K} combined results): {hits}")
    if total_entities > 0:
        logger.info(f"Recall@{RECALL_AT_K}: {recall_score:.4f}")
    else:
        logger.info("Recall not calculated as no entities were processed.")

    # 5. Clean up
    try:
        if hasattr(retriever, 'close') and callable(retriever.close):
            retriever.close()
            logger.info("Retriever resources closed.")
    except Exception as e:
        logger.error(f"Error closing retriever resources: {e}")

if __name__ == "__main__":
    eval_dir = os.path.dirname(EVALUATION_XML_FILE)
    if not os.path.exists(eval_dir):
        try:
            os.makedirs(eval_dir)
            logger.info(f"Created directory: {eval_dir}")
            logger.info(f"Please place '{os.path.basename(EVALUATION_XML_FILE)}' in {eval_dir} to run the evaluation.")
        except OSError as e:
            logger.error(f"Failed to create directory {eval_dir}: {e}")
            sys.exit(1) # Exit if cannot create data directory for eval file
    
    if not os.path.exists(EVALUATION_XML_FILE):
        logger.error(f"Evaluation XML file '{EVALUATION_XML_FILE}' not found. Please place it in the correct directory.")
    else:
        main()