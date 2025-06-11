# src/reranker/llm_reranker.py

import json
import logging
from typing import List, Dict, Optional, Tuple, Any

from sentence_transformers import CrossEncoder
import torch

# Configure logging for the module
logger = logging.getLogger(__name__)
# Basic configuration for the logger if no handlers are configured by the calling application
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LLMReranker:
    """
    Reranks documents using a CrossEncoder model.
    It takes a query and a list of candidate documents, computes a relevance score
    for each query-document pair, and returns the documents sorted by these scores.
    """

    def __init__(self,
                 model_name: str,
                 enriched_docs_path: str,
                 device: Optional[str] = None):
        """
        Initializes the LLMReranker.

        Args:
            model_name (str): The name of the CrossEncoder model to use
                              (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
            enriched_docs_path (str): Path to the JSON file containing enriched documents,
                                      expected to be a list of dicts with "id" and "text" keys.
            device (Optional[str]): The device to run the model on ("cuda", "cpu").
                                    If None, tries to use CUDA, otherwise CPU.
        """
        self.model_name = model_name
        self.enriched_docs_path = enriched_docs_path

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA available. Using CUDA for reranker.")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Using CPU for reranker.")
        
        try:
            self.model = CrossEncoder(self.model_name, device=self.device, trust_remote_code=True) # Added trust_remote_code for models like e5
            logger.info(f"Successfully loaded CrossEncoder model: {self.model_name} on {self.device}")
            model_max_length = self.model.tokenizer.model_max_length
            if model_max_length > 10000: # often indicates not properly set, e.g. 1e30
                model_max_length = 512 # Fallback to a common default
                logger.warning(f"Model tokenizer returned a very large max_length. Using fallback: {model_max_length}")
            self.model.max_length = model_max_length
            logger.info(f"Set CrossEncoder max_length to: {self.model.max_length}")

        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model {self.model_name}: {e}")
            if "intfloat/e5-mistral-7b-instruct" in self.model_name:
                 logger.warning(
                    f"Note: '{self.model_name}' is primarily an encoder model. "
                    "Using it directly with `CrossEncoder` might not yield optimal "
                    "results unless it's a variant fine-tuned for sequence-pair classification "
                    "or requires `trust_remote_code=True`. "
                    "Consider a bi-encoder approach (separate embeddings + cosine similarity) "
                    "or a prompt-based LLM reranking for this model type if standard CrossEncoder fails."
                )
            raise

        self.doc_texts = self._load_enriched_documents()
        if not self.doc_texts:
            logger.error(f"Enriched documents could not be loaded from {self.enriched_docs_path}. Reranker might not function correctly.")


    def _load_enriched_documents(self) -> Dict[str, str]:
        """
        Loads enriched documents and creates a mapping from document ID to text.

        Returns:
            Dict[str, str]: A dictionary mapping document CURIEs to their enriched text.
        """
        try:
            with open(self.enriched_docs_path, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            
            doc_map = {}
            for item in enriched_data:
                if "id" in item and "text" in item:
                    doc_map[item["id"]] = item["text"]
                else:
                    logger.warning(f"Skipping item due to missing 'id' or 'text' in {self.enriched_docs_path}: {item}")
            logger.info(f"Loaded {len(doc_map)} enriched documents for reranking from {self.enriched_docs_path}.")
            return doc_map
        except FileNotFoundError:
            logger.error(f"Enriched documents file not found: {self.enriched_docs_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.enriched_docs_path}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading enriched documents: {e}")
            return {}

    def _get_document_text_for_reranking(self, doc_id: str) -> Optional[str]:
        """
        Retrieves the pre-loaded enriched text for a given document ID.

        Args:
            doc_id (str): The CURIE of the document.

        Returns:
            Optional[str]: The enriched text of the document, or None if not found.
        """
        return self.doc_texts.get(doc_id)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            query (str): The search query.
            documents (List[Dict[str, Any]]): A list of document dictionaries retrieved
                                              from a previous stage (e.g., HybridRetriever).
                                              Each dict must contain an 'id' key.
            top_n (Optional[int]): The maximum number of documents to return after reranking.
                                   If None, all reranked documents are returned.

        Returns:
            List[Dict[str, Any]]: The list of documents, sorted by reranked scores,
                                  with a new 'rerank_score' key added to each document.
                                  Documents for which text could not be found are excluded.
        """
        if not query:
            logger.warning("Rerank called with an empty query. Returning original documents.")
            return documents[:top_n] if top_n is not None else documents
        
        if not documents:
            logger.info("Rerank called with no documents. Returning empty list.")
            return []

        if not self.doc_texts:
            logger.error("No enriched document texts loaded. Cannot perform reranking. Returning original documents.")
            return documents[:top_n] if top_n is not None else documents

        sentence_pairs: List[Tuple[str, str]] = []
        valid_documents_for_reranking: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                logger.warning(f"Document missing 'id' field, cannot rerank: {doc}")
                continue

            doc_text = self._get_document_text_for_reranking(doc_id)
            if doc_text:
                sentence_pairs.append((query, doc_text))
                valid_documents_for_reranking.append(doc)
            else:
                logger.warning(f"Could not find enriched text for document ID '{doc_id}'. Skipping for reranking.")
        
        if not sentence_pairs:
            logger.warning("No valid documents found to create sentence pairs for reranking. Returning original documents that were passed in.")
            return documents[:top_n] if top_n is not None else documents

        logger.info(f"Reranking {len(sentence_pairs)} document(s) for query: '{query}'")
        
        try:
            scores = self.model.predict(sentence_pairs, 
                                        show_progress_bar=False, # Set to True for verbose progress
                                        batch_size=32) # Adjust batch_size based on VRAM
        except Exception as e:
            logger.error(f"Error during CrossEncoder prediction: {e}")
            # Fallback: return original documents without reranking scores
            return documents[:top_n] if top_n is not None else documents

        # Add scores to documents and sort
        for i, doc in enumerate(valid_documents_for_reranking):
            doc["rerank_score"] = float(scores[i]) # Ensure score is float

        # Sort documents by rerank_score in descending order
        reranked_documents = sorted(valid_documents_for_reranking, key=lambda x: x.get("rerank_score", -float('inf')), reverse=True)

        if top_n is not None:
            reranked_documents = reranked_documents[:top_n]
            logger.info(f"Returning top {len(reranked_documents)} reranked documents.")
        else:
            logger.info(f"Returning all {len(reranked_documents)} reranked documents.")
            
        return reranked_documents


if __name__ == '__main__':
    import sys
    from pathlib import Path
    import random # For dummy retriever scores

    # --- Path Setup & Config Import ---
    # Add project root to sys.path to allow imports like src.config
    # Assumes this script is in onto_rag/src/reranker/
    PROJECT_ROOT_FOR_MAIN = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT_FOR_MAIN) not in sys.path:
        sys.path.append(str(PROJECT_ROOT_FOR_MAIN))

    try:
        from src.config import (
            ENRICHED_DOCUMENTS_FILE,
            RERANKER_MODEL_NAME,
            EMBEDDING_DEVICE,
            DEFAULT_RERANK_K,
            LOG_LEVEL, # Optional: if you want to use config's log level
            LOG_FILE   # Optional: if you want to use config's log file
        )
    except ImportError as e:
        print(f"Error importing from src.config: {e}")
        print("Please ensure that src/config.py exists and the script is run from the project root,")
        print("or that the PROJECT_ROOT_FOR_MAIN path is correctly set for your structure.")
        sys.exit(1)

    # --- Basic Logging Setup for the Example ---
    # Uses the main logger configured at the top of the file.
    # You can customize this further if needed, e.g., by setting level from config.LOG_LEVEL
    logger.setLevel(LOG_LEVEL if 'LOG_LEVEL' in locals() else "INFO")
    # Example of adding a file handler if LOG_FILE is configured:
    # if 'LOG_FILE' in locals() and LOG_FILE:
    #     fh = logging.FileHandler(LOG_FILE)
    #     fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    #     logger.addHandler(fh)
    #     logging.getLogger().addHandler(fh) # also add to root logger if basicConfig was called

    logger.info("--- Running LLMReranker Example with Real Data ---")
    logger.info(f"Using ENRICHED_DOCUMENTS_FILE: {ENRICHED_DOCUMENTS_FILE}")
    logger.info(f"Using RERANKER_MODEL_NAME: {RERANKER_MODEL_NAME}")
    logger.info(f"Using EMBEDDING_DEVICE: {EMBEDDING_DEVICE}")
    logger.info(f"Using DEFAULT_RERANK_K: {DEFAULT_RERANK_K}")


    # --- Load Sample Documents from Real Enriched Data ---
    sample_documents_from_retriever: List[Dict[str, Any]] = []
    try:
        with open(ENRICHED_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            all_enriched_docs = json.load(f)
        
        if not all_enriched_docs:
            logger.error(f"No documents found in {ENRICHED_DOCUMENTS_FILE}. Cannot proceed with the example.")
            sys.exit(1)

        # Select a few documents to simulate retriever output
        # Take up to 5, or fewer if the file has less
        num_docs_to_sample = min(len(all_enriched_docs), 5)
        if num_docs_to_sample == 0 :
             logger.error(f"The file {ENRICHED_DOCUMENTS_FILE} is empty. Cannot create sample documents.")
             sys.exit(1)
             
        for i in range(num_docs_to_sample):
            doc = all_enriched_docs[i]
            sample_documents_from_retriever.append({
                "id": doc.get("id"),
                "label": doc.get("label", "N/A"), # Get label if available
                "retriever_score": random.uniform(0.5, 1.0) # Dummy score
            })
        logger.info(f"Loaded {len(sample_documents_from_retriever)} sample documents for reranking.")

    except FileNotFoundError:
        logger.error(f"ERROR: The enriched documents file was not found: {ENRICHED_DOCUMENTS_FILE}")
        logger.error("Please ensure you have run the data ingestion and enrichment scripts first (e.g., `src.ingestion.enrich_documents.py`).")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"ERROR: Could not decode JSON from {ENRICHED_DOCUMENTS_FILE}. The file might be corrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading sample documents: {e}", exc_info=True)
        sys.exit(1)

    if not sample_documents_from_retriever:
        logger.error("No sample documents could be prepared. Exiting example.")
        sys.exit(1)
        
    # --- Reranker Initialization and Usage ---
    try:
        reranker = LLMReranker(
            model_name=RERANKER_MODEL_NAME,
            enriched_docs_path=str(ENRICHED_DOCUMENTS_FILE), # Ensure path is a string
            device=EMBEDDING_DEVICE
        )

        query = "Garlic" # Generic example, adjust to your data

        logger.info(f"\nOriginal sample documents (simulated retriever output) for query '{query}':")
        for doc in sample_documents_from_retriever:
            logger.info(f"  ID: {doc.get('id')}, Label: {doc.get('label')}, Retriever Score: {doc.get('retriever_score', 0.0):.4f}")

        reranked_results = reranker.rerank(query, sample_documents_from_retriever, top_n=DEFAULT_RERANK_K)

        logger.info(f"\nReranked documents (top {DEFAULT_RERANK_K} for query '{query}'):")
        if reranked_results:
            for doc in reranked_results:
                logger.info(f"  ID: {doc.get('id')}, Label: {doc.get('label')}, Rerank Score: {doc.get('rerank_score', 'N/A'):.4f}, Original Retriever Score: {doc.get('retriever_score', 'N/A')}")
        else:
            logger.info("  No results after reranking.")
        
        # Example of how you might switch to "intfloat/e5-mistral-7b-instruct"
        # Ensure it's set in your .env or config.py as RERANKER_MODEL_NAME
        # and that you have the resources for it.
        if RERANKER_MODEL_NAME != "intfloat/e5-mistral-7b-instruct" and False: # Set to True to test e5
            logger.info("\n--- Conceptual Test: Switching to intfloat/e5-mistral-7b-instruct ---")
            logger.warning("This is a large model and may require significant resources and download time.")
            logger.warning("Ensure 'intfloat/e5-mistral-7b-instruct' is configured as RERANKER_MODEL_NAME and 'trust_remote_code=True' might be needed.")
            
            try:
                e5_model_name = "intfloat/e5-mistral-7b-instruct"
                e5_reranker = LLMReranker(
                    model_name=e5_model_name,
                    enriched_docs_path=str(ENRICHED_DOCUMENTS_FILE),
                    device=EMBEDDING_DEVICE # "cuda" recommended for this model
                )
                e5_reranked_results = e5_reranker.rerank(query, sample_documents_from_retriever, top_n=DEFAULT_RERANK_K)
                logger.info(f"\nReranked documents with {e5_model_name} (top {DEFAULT_RERANK_K} for query '{query}'):")
                if e5_reranked_results:
                    for doc in e5_reranked_results:
                        logger.info(f"  ID: {doc.get('id')}, Label: {doc.get('label')}, Rerank Score: {doc.get('rerank_score', 'N/A'):.4f}")
                else:
                    logger.info("  No results after reranking with E5.")
            except Exception as e_e5:
                logger.error(f"Could not initialize or use {e5_model_name} reranker: {e_e5}", exc_info=True)
                logger.warning(f"Skipping {e5_model_name} reranking part of the example.")


    except Exception as e:
        logger.error(f"An error occurred in the main example: {e}", exc_info=True)

    logger.info("--- LLMReranker Example Finished ---")