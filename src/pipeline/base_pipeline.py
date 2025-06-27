# src/pipeline/base_pipeline.py
import sys
import logging
from typing import List, Dict, Any, Optional, Type
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_selectors.base_selector import BaseSelector
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.llm_reranker import LLMReranker
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseRAGPipeline:
    """
    A base class for the RAG pipeline, abstracting the common logic for
    retrieving, reranking, and selecting ontology terms.
    
    Subclasses should specify the selector to use.
    """
    def __init__(self, selector_class: Type[BaseSelector]):
        """
        Initializes the RAG Pipeline.

        Args:
            selector_class (Type[BaseSelector]): The class of the selector to use
                                                 (e.g., GeminiSelector, OllamaSelector).
        """
        logger.info(f"Initializing RAG Pipeline with {selector_class.__name__}...")
        try:
            self.retriever = HybridRetriever()

            all_enriched_docs_paths = [
                data['enriched_docs_path'] 
                for data in config.ONTOLOGIES_CONFIG.values()
                if data['enriched_docs_path'].exists()
            ]

            if not all_enriched_docs_paths:
                raise FileNotFoundError("No enriched document files found for any configured ontology. Please run the ingestion pipeline.")

            logger.info(f"Initializing LLMReranker with {len(all_enriched_docs_paths)} enriched document file(s).")
            
            self.reranker = LLMReranker(
                model_name=config.RERANKER_MODEL_NAME,
                enriched_docs_paths=all_enriched_docs_paths,
                device=config.EMBEDDING_DEVICE
            )
            
            self.selector = selector_class(retriever=self.retriever)
            logger.info("RAG Pipeline initialized successfully.")
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.error("Please run ingestion scripts (e.g., 'scripts/rebuild_base.bash') and ensure necessary API keys are set in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during pipeline initialization: {e}", exc_info=True)
            raise

    def run(self, 
            query: str, 
            lexical_k: int = config.DEFAULT_K_LEXICAL, 
            vector_k: int = config.DEFAULT_K_VECTOR, 
            rerank_top_n: int = 10
            ) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Executes the full pipeline for a given query.

        Returns:
            A tuple containing (final_result_dict, candidates_list), or None.
            The final_result_dict includes the confidence score and reasoning.
        """
        logger.info("Running pipeline for query: '%s'", query)

        # 1. Retrieve
        retriever_output = self.retriever.search(query, lexical_limit=lexical_k, vector_k=vector_k)
        lexical_results = retriever_output.get("lexical_results", [])
        vector_results = retriever_output.get("vector_results", [])
        
        # 2. Merge unique candidates
        combined_candidates = []
        seen_ids = set()
        for doc in lexical_results + vector_results:
            doc_id = doc.get('id')
            if doc_id and doc_id not in seen_ids:
                combined_candidates.append(doc)
                seen_ids.add(doc_id)
        
        if not combined_candidates:
            logger.warning("No candidates found for query: '%s'", query)
            return None

        # 3. Rerank
        reranked_candidates = self.reranker.rerank(query, combined_candidates, top_n=rerank_top_n)

        if not reranked_candidates:
            logger.warning("No candidates left after reranking for query: '%s'", query)
            return None
        
        logger.info(f"Top {len(reranked_candidates)} candidates after reranking passed to LLM selector.")

        # 4. Select with LLM
        selection = self.selector.select_best_term(query, reranked_candidates)

        # 5. Process selection
        if not selection:
            logger.warning("LLM selection failed. Returning the top reranked result as a fallback.")
            top_fallback = reranked_candidates[0]
            chosen_term_details = self.retriever.get_term_details(top_fallback['id'])
            if chosen_term_details:
                chosen_term_details['confidence_score'] = 0.0
                chosen_term_details['explanation'] = "FALLBACK: LLM selection failed. This is the top result from the reranker."
            return chosen_term_details, reranked_candidates

        chosen_id = selection['chosen_id']
        if chosen_id in ('0', '-1'):
            logger.info("LLM determined no suitable match exists for query: '%s'", query)
            no_match_result = {
                'id': chosen_id,
                'label': 'No Match Found',
                'definition': 'The Language Model determined that no candidate was a suitable match for the query.',
                'synonyms': [], 'parents': [], 'ancestors': [], 'relations': {},
                'confidence_score': selection.get('confidence_score', 0.0),
                'explanation': selection.get('explanation', 'No explanation provided.')
            }
            return no_match_result, reranked_candidates
        
        chosen_term_details = self.retriever.get_term_details(chosen_id)
        if not chosen_term_details:
            logger.error("LLM chose ID '%s', but its details could not be retrieved.", chosen_id)
            return None

        # Add the confidence and explanation from the selector to the final result
        chosen_term_details['confidence_score'] = selection.get('confidence_score', 0.0)
        chosen_term_details['explanation'] = selection.get('explanation', 'No explanation provided.')
        
        return chosen_term_details, reranked_candidates

    def close(self):
        """Closes any open resources, like database connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")