import logging
import json
from typing import List, Dict, Any, Optional
import asyncio

from src import config
from src.components.retriever import HybridRetriever
from src.components.selector import Selector
from src.components.scorer import ConfidenceScorer
from src.components.synonyms import SynonymGenerator

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Simple RAG pipeline composed from injected components."""

    def __init__(
        self,
        retriever: HybridRetriever,
        selector: Selector,
        confidence_scorer: ConfidenceScorer,
        synonym_generator: Optional[SynonymGenerator] = None,
    ) -> None:
        self.retriever = retriever
        self.selector = selector
        self.confidence_scorer = confidence_scorer
        self.synonym_generator = synonym_generator
        logger.info("RAG Pipeline initialized successfully.")

    async def run(
            self,
            query: str,
            context: Optional[str] = None,
            lexical_k: int = config.DEFAULT_K_LEXICAL,
            vector_k: int = config.DEFAULT_K_VECTOR,
            target_ontologies: Optional[List[str]] = None,
            semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        
        if semaphore:
            await semaphore.acquire()
        
        try:
            queries_to_try = [query]
            queries_tried = set()
            best_result_so_far = None
            best_candidates_so_far: list[dict] = []
            loop_count = 0
            last_feedback = ""

            while queries_to_try and loop_count < config.MAX_PIPELINE_LOOPS:
                current_query = queries_to_try.pop(0)
                if current_query in queries_tried:
                    continue
                
                loop_count += 1
                queries_tried.add(current_query)
                logger.info(f"--- Starting Pipeline Attempt {loop_count}/{config.MAX_PIPELINE_LOOPS} for query: '{current_query}' ---")

                effective_targets = target_ontologies if target_ontologies is not None else getattr(config, "RESTRICT_TARGET_ONTOLOGIES", None)
                retriever_output = self.retriever.search(
                    current_query,
                    lexical_limit=lexical_k,
                    vector_k=vector_k,
                    target_ontologies=effective_targets,
                )
                candidates = retriever_output.get("lexical_results", []) + retriever_output.get("vector_results", [])

                if not candidates:
                    logger.warning(f"No candidates found for query '{current_query}'.")
                    continue

                logger.info(f"{len(candidates)} candidates passed to LLM selector for query '{current_query}'.")
                
                score = None
                confidence_result = None
                
                selection = await self.selector.select_best_term(
                    current_query,
                    candidates,
                    context=context if 'context' in locals() else "",
                    feedback=last_feedback,
                )

                if not selection or selection['chosen_id'] in ('0', '-1'):
                    logger.info(f"Selector found no suitable match for '{current_query}' among the candidates.")
                    current_result = {
                        'id': None,
                        'confidence_score': 0.0,
                        'selector_explanation': selection.get('selector_explanation') if selection else 'Selector returned no valid selection.'
                    }
                    score = 0.0
                    last_feedback = current_result['selector_explanation']
                else:
                    chosen_id = selection['chosen_id']
                    chosen_term_details = self.retriever.get_term_details(chosen_id)
                    if not chosen_term_details:
                        logger.error(f"Selector chose ID '{chosen_id}', but its details could not be retrieved.")
                        continue 

                    logger.info(f"Scoring selection '{chosen_id}' against the ORIGINAL query: '{query}'")

                    try:
                        confidence_result = await self.confidence_scorer.score_confidence(
                            query=query,
                            chosen_term_details=chosen_term_details,
                            all_candidates=candidates,
                            context=context or ""
                        )
                    except Exception as e:
                        logger.error("Confidence scorer failed: %s", e, exc_info=True)
                        confidence_result = None

                    current_result = chosen_term_details
                    current_result['selector_explanation'] = selection.get('selector_explanation', 'No explanation available.')
                    
                    if confidence_result is not None:
                        score = confidence_result.get('confidence_score', None)
                        current_result['confidence_score'] = score
                        current_result['scorer_explanation'] = confidence_result.get('scorer_explanation', 'No explanation available.')
                        current_result['suggested_alternatives'] = confidence_result.get('suggested_alternatives', [])
                        last_feedback = current_result['scorer_explanation']
                    else:
                        score = -1.0
                        current_result['confidence_score'] = score
                        current_result['scorer_explanation'] = 'Scorer failed to provide explanation.'
                        current_result['suggested_alternatives'] = []
                        last_feedback = current_result['scorer_explanation']

                logger.info(f"Selection Details: Label: '{current_result.get('label', 'N/A')}', ID: {current_result.get('id', 'N/A')}, Confidence: {current_result.get('confidence_score', 0.0):.2f}")

                if best_result_so_far is None or current_result.get('confidence_score', 0.0) > best_result_so_far.get('confidence_score', 0.0):
                    best_result_so_far = current_result
                    best_candidates_so_far = candidates

                if score is not None and score >= config.CONFIDENCE_THRESHOLD:
                    logger.info("Confidence above threshold; stopping.")
                    return current_result, candidates

                suggestions = []
                if confidence_result:
                    raw = confidence_result.get('suggested_alternatives') or []
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                            raw = parsed if isinstance(parsed, list) else [raw]
                        except Exception:
                            raw = [s.strip() for s in raw.strip('[]').split(',') if s.strip()]
                    suggestions = [s for s in raw if isinstance(s, str) and s.strip()]

                if suggestions:
                    already = set(queries_tried) | set(queries_to_try)
                    to_add = [s for s in suggestions if s not in already and s != query]
                    if to_add:
                        logger.info("Scorer suggested alternatives (pre-synonyms): %s", to_add)
                        queries_to_try = to_add + queries_to_try

                if self.synonym_generator and not suggestions and not queries_to_try:
                    try:
                        syns = await self.synonym_generator.generate_synonyms(
                            query=query,
                            context=context or "",
                            feedback=last_feedback,
                        )
                        if syns:
                            already = set(queries_tried) | set(queries_to_try)
                            syns = [s for s in syns if s and s not in already and s != query]
                            if syns:
                                logger.info("Generated synonyms for retry: %s", syns)
                                queries_to_try.extend(syns)
                    except Exception as e:
                        logger.error("Synonym generator failed: %s", e, exc_info=True)

            return best_result_so_far, best_candidates_so_far

        finally:
            if semaphore:
                semaphore.release()


    def close(self):
        """Closes any open resources, like database connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")


def create_pipeline(provider: str = "gemini") -> RAGPipeline:
    """Creates/Instantiates the pipeline with Gemini components."""
    # Provider arg is kept for compatibility but ignored/checked since we only support 'gemini' in this simple version
    if provider != "gemini":
         logger.warning(f"Provider '{provider}' requested, but only 'gemini' is supported in this simplified version. Using Gemini.")
    
    retriever = HybridRetriever()
    selector = Selector(retriever)
    confidence = ConfidenceScorer()
    synonym = SynonymGenerator()

    return RAGPipeline(
        retriever=retriever,
        selector=selector,
        confidence_scorer=confidence,
        synonym_generator=synonym,
    )
