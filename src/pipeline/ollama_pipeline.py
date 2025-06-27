# src/pipeline/pipeline.py
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.llm_reranker import LLMReranker
from src.rag_selectors.ollama_selector import OllamaSelector
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        try:
            self.retriever = HybridRetriever()

            all_enriched_docs_paths = [
                data['enriched_docs_path'] 
                for data in config.ONTOLOGIES_CONFIG.values()
                if os.path.exists(data['enriched_docs_path']) 
            ]

            if not all_enriched_docs_paths:
                raise FileNotFoundError("No enriched document files found for any configured ontology. Please run the ingestion pipeline.")

            logger.info(f"Initializing LLMReranker with {len(all_enriched_docs_paths)} enriched document file(s).")
            
            self.reranker = LLMReranker(
                model_name=config.RERANKER_MODEL_NAME,
                enriched_docs_paths=all_enriched_docs_paths,  # Pass the list of paths
                device=config.EMBEDDING_DEVICE
            )
            self.selector = OllamaSelector(retriever=self.retriever) # <--- UPDATED INSTANTIATION
            logger.info("RAG Pipeline initialized successfully.")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.error("Please run 'scripts/rebuild_base.bash' and ensure GEMINI_API_KEY is set in your .env file.")
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
        
        # 2. Merge
        combined_candidates = []
        seen_ids = set()
        for doc in lexical_results + vector_results:
            doc_id = doc.get('id')
            if doc_id and doc_id not in seen_ids:
                combined_candidates.append(doc)
                seen_ids.add(doc_id)
        
        if not combined_candidates:
            logger.warning("No candidates found.")
            return None

        # 3. Rerank
        reranked_candidates = self.reranker.rerank(query, combined_candidates, top_n=rerank_top_n)

        if not reranked_candidates:
            logger.warning("No candidates left after reranking.")
            return None
        
        logger.info(f"Top {len(reranked_candidates)} candidates after reranking passed to LLM selector.")

        # 4. Select with LLM
        selection = self.selector.select_best_term(query, reranked_candidates)

        if not selection:
            logger.error("LLM selection failed. Returning the top reranked result as a fallback.")
            top_fallback = reranked_candidates[0]
            chosen_term_details = self.retriever.get_term_details(top_fallback['id'])
            chosen_term_details['explanation'] = "FALLBACK: LLM selection failed. This is the top result from the reranker."
            return chosen_term_details, reranked_candidates

        # 5. Get final details and return
        chosen_id = selection['chosen_id']
        if chosen_id == '0' or chosen_id == '-1':
            logger.info("LLM selected no suitable match. Returning No match.")
            no_match_result = {
                'id': chosen_id,  # Preserve the -1 or 0 ID as the signal
                'label': 'No Match Found',
                'definition': 'The Language Model determined that no candidate was a suitable match for the query.',
                'synonyms': [],
                'parents': [],
                'ancestors': [],
                'relations': {},
                'confidence_score': selection.get('confidence_score', 0.0),
                'explanation': selection.get('explanation', 'No explanation provided.')
            }
            return no_match_result, reranked_candidates
        chosen_term_details = self.retriever.get_term_details(chosen_id)
        
        if not chosen_term_details:
            logger.error("LLM chose ID '%s', but its details could not be retrieved.", chosen_id)
            return None

        chosen_term_details['confidence_score'] = selection.get('confidence_score')
        chosen_term_details['explanation'] = selection['explanation']
        return chosen_term_details, reranked_candidates

    def close(self):
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")