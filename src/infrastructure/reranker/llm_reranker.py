# infrastructure/reranker/llm_reranker.py

import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Union

from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

class LLMReranker:
    """
    Reranks documents using a CrossEncoder model. It loads enriched documents
    from one or more files and uses them to rerank candidate documents for a given query.
    """

    def __init__(self,
                 model_name: str,
                 enriched_docs_paths: Union[str, List[str]],
                 device: Optional[str] = None):
        """
        Initializes the LLMReranker.

        Args:
            model_name (str): The name of the CrossEncoder model to use.
            enriched_docs_paths (Union[str, List[str]]): A single path or a list of paths
                                                         to JSON files containing enriched documents.
            device (Optional[str]): The device to run the model on ("cuda", "cpu").
        """
        self.model_name = model_name
        self.enriched_docs_paths = enriched_docs_paths
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            logger.info(f"Loading CrossEncoder model: {self.model_name} on {self.device}")
            self.model = CrossEncoder(self.model_name, device=self.device, trust_remote_code=True)
            model_max_length = getattr(self.model.tokenizer, 'model_max_length', 512)
            if model_max_length > 10000: # Handle unrealistic default values
                model_max_length = 512
            self.model.max_length = model_max_length
            logger.info(f"Set CrossEncoder max_length to: {self.model.max_length}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model {self.model_name}: {e}", exc_info=True)
            raise

        self.doc_texts = self._load_enriched_documents()
        if not self.doc_texts:
            logger.error(f"No documents loaded from {self.enriched_docs_paths}. Reranker will be ineffective.")

    def _load_enriched_documents(self) -> Dict[str, str]:
        """
        Loads enriched documents from one or more files and creates a single
        mapping from document ID to text.

        Returns:
            Dict[str, str]: A dictionary mapping document CURIEs to their enriched text.
        """
        doc_map = {}
        paths_to_load = self.enriched_docs_paths
        if isinstance(paths_to_load, str):
            paths_to_load = [paths_to_load]

        for path in paths_to_load:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    enriched_data = json.load(f)
                
                for item in enriched_data:
                    if "id" in item and "text" in item:
                        doc_map[item["id"]] = item["text"]
            except FileNotFoundError:
                logger.error(f"Enriched documents file not found: {path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {path}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading {path}: {e}", exc_info=True)
        
        logger.info(f"Loaded a total of {len(doc_map)} enriched documents from {len(paths_to_load)} file(s).")
        return doc_map

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.
        (This method requires no changes as it uses the pre-loaded self.doc_texts map).
        """
        if not query or not documents:
            return documents[:top_n] if top_n is not None else documents

        if not self.doc_texts:
            logger.error("No enriched document texts loaded. Cannot perform reranking.")
            return documents[:top_n] if top_n is not None else documents

        sentence_pairs: List[Tuple[str, str]] = []
        valid_documents_for_reranking: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("id")
            if doc_id:
                doc_text = self.doc_texts.get(doc_id)
                if doc_text:
                    sentence_pairs.append((query, doc_text))
                    valid_documents_for_reranking.append(doc)
                else:
                    logger.warning(f"Could not find enriched text for document ID '{doc_id}'. Skipping for reranking.")
        
        if not sentence_pairs:
            logger.warning("No valid document texts found for the given candidates.")
            return []

        logger.info(f"Reranking {len(sentence_pairs)} document(s) for query: '{query}'")
        scores = self.model.predict(sentence_pairs, show_progress_bar=False, batch_size=32)

        for i, doc in enumerate(valid_documents_for_reranking):
            doc["rerank_score"] = float(scores[i])

        reranked_documents = sorted(valid_documents_for_reranking, key=lambda x: x.get("rerank_score", -float('inf')), reverse=True)

        return reranked_documents[:top_n] if top_n is not None else reranked_documents
