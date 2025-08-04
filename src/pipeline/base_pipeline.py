# src/pipeline/base_pipeline.py
import sys
import logging
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import asyncio


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_selectors.base_selector import BaseSelector
from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.llm_reranker import LLMReranker
from src import config # <-- Ensure config is imported to get loop constants
from src.synonym_generators.ollama_synonym_generator import OllamaSynonymGenerator
from src.synonym_generators.gemini_synonym_generator import GeminiSynonymGenerator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseRAGPipeline:
    """
    A base class for the RAG pipeline, abstracting the common logic for
    retrieving, reranking, and selecting ontology terms.

    Subclasses should specify the selector to use.
    """
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

            '''all_enriched_docs_paths = [
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
            )'''

            self.selector = selector_class(retriever=self.retriever)
            self.confidence_scorer = confidence_scorer_class()

            # --- Initialize Synonym Generator ---
            if config.PIPELINE == "ollama":
                from src.synonym_generators.ollama_synonym_generator import OllamaSynonymGenerator
                self.synonym_generator = OllamaSynonymGenerator()
            elif config.PIPELINE == "gemini":
                from src.synonym_generators.gemini_synonym_generator import GeminiSynonymGenerator
                self.synonym_generator = GeminiSynonymGenerator()
            elif config.PIPELINE == "huggingface":
                # This is the new part
                from src.synonym_generators.huggingface_synonym_generator import HuggingFaceSynonymGenerator
                self.synonym_generator = HuggingFaceSynonymGenerator()
            else:
                self.synonym_generator = None
                logger.warning("No synonym generator configured for the current pipeline.")

            logger.info("RAG Pipeline initialized successfully.")

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.error("Please run ingestion scripts (e.g., 'scripts/rebuild_base.bash') and ensure necessary API keys are set in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during pipeline initialization: {e}", exc_info=True)
            raise

    # In src/pipeline/base_pipeline.py

    async def run(self,
            query: str,
            lexical_k: int = config.DEFAULT_K_LEXICAL,
            vector_k: int = config.DEFAULT_K_VECTOR,
            semaphore: Optional[asyncio.Semaphore] = None
            ) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Executes the pipeline with per-query validation, but scores
        confidence against the original user query for a final check.
        """
        # ADDED: Acquire semaphore if provided
        if semaphore:
            await semaphore.acquire()
        
        try:
            queries_to_try = [query]
            queries_tried = set()
            best_result_so_far = None
            loop_count = 0

            # The 'query' variable will always hold the original user query.
            # The 'current_query' variable will change with each loop.

            while queries_to_try and loop_count < config.MAX_PIPELINE_LOOPS:
                current_query = queries_to_try.pop(0)
                if current_query in queries_tried:
                    continue
                
                loop_count += 1
                queries_tried.add(current_query)
                logger.info(f"--- Starting Pipeline Attempt {loop_count}/{config.MAX_PIPELINE_LOOPS} for query: '{current_query}' ---")

                # 1. Retrieve candidates for the CURRENT query (e.g., "Brilliant Blue FCF")
                retriever_output = self.retriever.search(current_query, lexical_limit=lexical_k, vector_k=vector_k)
                candidates = retriever_output.get("lexical_results", []) + retriever_output.get("vector_results", [])

                if not candidates:
                    logger.warning(f"No candidates found for query '{current_query}'.")
                    continue

                # 2. Select the best term using the CURRENT query for context.
                # This is the "Focused Selection" step.
                logger.info(f"{len(candidates)} candidates passed to LLM selector for query '{current_query}'.")
                selection = await self.selector.select_best_term(current_query, candidates) # CHANGED: Added await

                if not selection or selection['chosen_id'] in ('0', '-1'):
                    logger.warning(f"Selector failed for query '{current_query}'.")
                    continue
                
                chosen_id = selection['chosen_id']
                chosen_term_details = self.retriever.get_term_details(chosen_id)
                if not chosen_term_details:
                    logger.error(f"Selector chose ID '{chosen_id}', but its details could not be retrieved.")
                    continue

                # --- THIS IS THE MODIFIED SECTION ---
                # 3. Score confidence using the ORIGINAL query for context.
                # This is the "Holistic Scoring" step.
                logger.info(f"Scoring selection '{chosen_id}' against the ORIGINAL query: '{query}'")

                confidence_result = await self.confidence_scorer.score_confidence( # CHANGED: Added await
                    query=query, # <-- CRUCIAL CHANGE: Use the original 'query' variable here
                    chosen_term_details=chosen_term_details,
                    all_candidates=candidates
                )
                # ------------------------------------
                
                current_result = chosen_term_details
                if confidence_result:
                    current_result['confidence_score'] = confidence_result.get('confidence_score', 0.0)
                    current_result['explanation'] = confidence_result.get('explanation', selection.get('explanation'))
                else:
                    current_result['confidence_score'] = 0.0
                    current_result['explanation'] = selection.get('explanation')

                logger.info(f"Selection: '{current_result.get('label')}' (ID: {chosen_id}), Confidence against original query: {current_result['confidence_score']:.2f}, Explanation: {current_result['explanation']}")

                # ... (Rest of the loop: update best result, check threshold, generate synonyms) ...
                if best_result_so_far is None or current_result['confidence_score'] > best_result_so_far.get('confidence_score', 0.0):
                    best_result_so_far = current_result

                if best_result_so_far and best_result_so_far.get('confidence_score', 0.0) >= config.CONFIDENCE_THRESHOLD:
                    return best_result_so_far, candidates

                if self.synonym_generator and current_query == query and not queries_to_try:
                    synonyms = await self.synonym_generator.generate_synonyms(query) # CHANGED: Added await
                    if synonyms:
                        queries_to_try.extend(synonyms)

            return best_result_so_far, []

        finally:
            # ADDED: Release semaphore if provided
            if semaphore:
                semaphore.release()


    def close(self):
        """Closes any open resources, like database connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")