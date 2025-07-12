# src/pipeline/base_pipeline.py
import sys
import logging
from typing import List, Dict, Any, Optional, Type
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_selectors.base_selector import BaseSelector
from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
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
    # MODIFIED __init__ to accept a confidence scorer
    def __init__(self, selector_class: Type[BaseSelector], confidence_scorer_class: Type[BaseConfidenceScorer]):
        """
        Initializes the RAG Pipeline.

        Args:
            selector_class (Type[BaseSelector]): The class of the selector to use.
            confidence_scorer_class (Type[BaseConfidenceScorer]): The class of the confidence scorer to use.
        """
        logger.info(f"Initializing RAG Pipeline with {selector_class.__name__} and {confidence_scorer_class.__name__}...")
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
            self.confidence_scorer = confidence_scorer_class() # ADDED
            logger.info("RAG Pipeline initialized successfully.")
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.error("Please run ingestion scripts (e.g., 'scripts/rebuild_base.bash') and ensure necessary API keys are set in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during pipeline initialization: {e}", exc_info=True)
            raise

    # MODIFIED run method to incorporate the two-step selection and scoring
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
        combined_candidates_dict = {
            doc['id']: doc 
            for doc in lexical_results + vector_results 
            if doc.get('id')
        }
        combined_candidates = list(combined_candidates_dict.values())
        
        if not combined_candidates:
            logger.warning("No candidates found for query: '%s'", query)
            return None

        # 3. Rerank (Uncommented as per intended logic)
        reranked_candidates = self.reranker.rerank(query, combined_candidates, top_n=rerank_top_n)

        if not reranked_candidates:
            logger.warning("No candidates left after reranking for query: '%s'", query)
            return None
        
        logger.info(f"Top {len(reranked_candidates)} candidates after reranking passed to LLM selector.")

        # 4. Select with Selector Agent
        selection = self.selector.select_best_term(query, reranked_candidates)

        if not selection:
            logger.warning("LLM selection failed. Cannot proceed to confidence scoring.")
            return None, reranked_candidates

        chosen_id = selection['chosen_id']
        initial_explanation = selection.get('explanation', 'No explanation provided.')
        
        if chosen_id in ('0', '-1'):
            logger.info("Selector determined no suitable match exists for query: '%s'", query)
            no_match_result = {
                'id': chosen_id,
                'label': 'No Match Found',
                'definition': 'The selector model determined that no candidate was a suitable match.',
                'synonyms': [], 'parents': [], 'ancestors': [], 'relations': {},
                'confidence_score': 0.0,
                'explanation': initial_explanation
            }
            return no_match_result, reranked_candidates

        chosen_term_details = self.retriever.get_term_details(chosen_id)
        if not chosen_term_details:
            logger.error("Selector chose ID '%s', but its details could not be retrieved.", chosen_id)
            return None, reranked_candidates
        
        # 5. Score Confidence with Confidence Agent
        logger.info("Passing selection to confidence scorer for query: '%s'", query)
        confidence_result = self.confidence_scorer.score_confidence(
            query=query,
            chosen_term_details=chosen_term_details,
            all_candidates=reranked_candidates
        )

        if confidence_result:
            chosen_term_details['confidence_score'] = confidence_result.get('confidence_score', 0.0)
            chosen_term_details['explanation'] = confidence_result.get('explanation', initial_explanation)
            logger.info(f"Confidence score for '{chosen_id}': {chosen_term_details['confidence_score']:.2f}")
        else:
            logger.warning("Confidence scoring failed. Using default values.")
            chosen_term_details['confidence_score'] = 0.0
            chosen_term_details['explanation'] = initial_explanation
        
        return chosen_term_details, reranked_candidates

    def close(self):
        """Closes any open resources, like database connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")