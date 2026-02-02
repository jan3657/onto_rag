import logging
import json
from typing import List, Dict, Any, Optional
import asyncio

from src import config
from src.components.retriever import HybridRetriever
from src.components.selector import Selector
from src.components.scorer import ConfidenceScorer
from src.components.synonyms import SynonymGenerator
from src.utils.tracing import generate_trace_id, trace_log

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
        """
        Run the RAG pipeline with full traceability.
        
        Returns:
            Tuple of (best_result, candidates) or None
        """
        # Generate unique trace ID for this query
        trace_id = generate_trace_id()
        
        trace_log("pipeline_start", trace_id, query, 
                  context=context or "", target_ontologies=target_ontologies)
        
        # Use async context manager for semaphore to prevent deadlock
        async def _run_with_semaphore():
            queries_to_try = [query]
            queries_tried = set()
            best_result_so_far = None
            best_candidates_so_far: list[dict] = []
            attempt_index = 0
            last_feedback = ""

            while queries_to_try and attempt_index < config.MAX_PIPELINE_LOOPS:
                current_query = queries_to_try.pop(0)
                if current_query in queries_tried:
                    continue
                
                queries_tried.add(current_query)
                attempt_index += 1
                
                trace_log("attempt_start", trace_id, query, current_query, attempt_index,
                          queries_remaining=len(queries_to_try), queries_tried=list(queries_tried))

                # --- RETRIEVAL ---
                trace_log("retrieval_start", trace_id, query, current_query, attempt_index)
                
                effective_targets = target_ontologies if target_ontologies is not None else getattr(config, "RESTRICT_TARGET_ONTOLOGIES", None)
                retriever_output = self.retriever.search(
                    current_query,
                    lexical_limit=lexical_k,
                    vector_k=vector_k,
                    target_ontologies=effective_targets,
                    trace_id=trace_id,
                )
                
                # Use merged candidates (deduplicated)
                candidates = retriever_output.get("merged_candidates", [])
                merge_stats = retriever_output.get("merge_stats", {})
                
                trace_log("retrieval_complete", trace_id, query, current_query, attempt_index,
                          candidate_count=len(candidates), merge_stats=merge_stats)

                if not candidates:
                    trace_log("retry_triggered", trace_id, query, current_query, attempt_index,
                              reason="no_candidates", message="No candidates found for query")
                    continue

                # --- SELECTION ---
                trace_log("selection_start", trace_id, query, current_query, attempt_index,
                          candidate_count=len(candidates))
                
                selection = await self.selector.select_best_term(
                    current_query,
                    candidates,
                    context=context or "",
                    feedback=last_feedback,
                    trace_id=trace_id,
                )

                # Check for no-match from selector
                if not selection or selection.get('chosen_id') in ('0', '-1', None):
                    explanation = selection.get('selector_explanation', 'Selector returned no valid selection.') if selection else 'Selector returned None'
                    
                    trace_log("retry_triggered", trace_id, query, current_query, attempt_index,
                              reason="selector_no_match", 
                              explanation=explanation,
                              chosen_id=selection.get('chosen_id') if selection else None)
                    
                    current_result = {
                        'id': None,
                        'confidence_score': 0.0,
                        'selector_explanation': explanation
                    }
                    last_feedback = explanation
                    
                    # Update best if this is better
                    if best_result_so_far is None:
                        best_result_so_far = current_result
                        best_candidates_so_far = candidates
                    
                    # Try synonym generation if no more queries
                    if self.synonym_generator and not queries_to_try:
                        await self._try_synonym_generation(
                            query, context, last_feedback, queries_tried, queries_to_try, trace_id, attempt_index
                        )
                    continue

                chosen_id = selection['chosen_id']
                trace_log("selection_complete", trace_id, query, current_query, attempt_index,
                          chosen_id=chosen_id, 
                          explanation=selection.get('selector_explanation', ''))

                # --- VALIDATE CHOSEN ID ---
                chosen_term_details = self.retriever.get_term_details(chosen_id)
                if not chosen_term_details:
                    source_idx = next((i for i, c in enumerate(candidates) if c.get('id') == chosen_id), None)
                    source_candidate = candidates[source_idx] if source_idx is not None else None
                    
                    trace_log("data_integrity_error", trace_id, query, current_query, attempt_index,
                              chosen_id=chosen_id,
                              candidate_index=source_idx,
                              source=source_candidate.get('source') if source_candidate else 'unknown',
                              source_ontology=source_candidate.get('source_ontology') if source_candidate else 'unknown',
                              message="Chosen ID not found in ontology metadata")
                    
                    logger.error(
                        f"[DATA_INTEGRITY_ERROR] Selector chose ID '{chosen_id}' (candidate_index={source_idx}) "
                        f"but its details could not be retrieved from ontology metadata lookup."
                    )
                    continue 

                # --- SCORING ---
                trace_log("scoring_start", trace_id, query, current_query, attempt_index,
                          chosen_id=chosen_id)

                try:
                    confidence_result = await self.confidence_scorer.score_confidence(
                        query=query,  # Score against ORIGINAL query
                        chosen_term_details=chosen_term_details,
                        all_candidates=candidates,
                        context=context or "",
                        trace_id=trace_id,
                    )
                except Exception as e:
                    logger.error("Confidence scorer failed: %s", e, exc_info=True)
                    trace_log("scoring_error", trace_id, query, current_query, attempt_index,
                              error=str(e))
                    confidence_result = None

                # Build result
                current_result = dict(chosen_term_details)
                current_result['selector_explanation'] = selection.get('selector_explanation', 'No explanation available.')
                
                if confidence_result is not None:
                    score = confidence_result.get('confidence_score', 0.0)
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

                trace_log("scoring_complete", trace_id, query, current_query, attempt_index,
                          chosen_id=chosen_id,
                          confidence_score=score,
                          explanation=current_result.get('scorer_explanation', ''))

                # Update best result
                if best_result_so_far is None or current_result.get('confidence_score', 0.0) > best_result_so_far.get('confidence_score', 0.0):
                    best_result_so_far = current_result
                    best_candidates_so_far = candidates

                # --- CHECK ACCEPTANCE CRITERIA ---
                if score >= config.CONFIDENCE_THRESHOLD:
                    trace_log("accepted", trace_id, query, current_query, attempt_index,
                              chosen_id=chosen_id,
                              confidence_score=score,
                              reason="above_confidence_threshold",
                              threshold=config.CONFIDENCE_THRESHOLD)
                    return current_result, candidates

                # Check if below minimum confidence
                if score < config.MIN_CONFIDENCE:
                    trace_log("retry_triggered", trace_id, query, current_query, attempt_index,
                              reason="below_min_confidence",
                              confidence_score=score,
                              min_confidence=config.MIN_CONFIDENCE,
                              explanation=last_feedback)
                else:
                    # Between MIN_CONFIDENCE and CONFIDENCE_THRESHOLD
                    trace_log("retry_triggered", trace_id, query, current_query, attempt_index,
                              reason="below_confidence_threshold",
                              confidence_score=score,
                              threshold=config.CONFIDENCE_THRESHOLD,
                              explanation=last_feedback)

                # --- TRY ALTERNATIVES FROM SCORER ---
                suggestions = self._parse_suggestions(confidence_result)
                if suggestions:
                    already = set(queries_tried) | set(queries_to_try)
                    to_add = [s for s in suggestions if s not in already and s != query]
                    if to_add:
                        trace_log("alternatives_added", trace_id, query, current_query, attempt_index,
                                  source="scorer", alternatives=to_add)
                        queries_to_try = to_add + queries_to_try

                # --- TRY SYNONYM GENERATION ---
                if self.synonym_generator and not suggestions and not queries_to_try:
                    await self._try_synonym_generation(
                        query, context, last_feedback, queries_tried, queries_to_try, trace_id, attempt_index
                    )

            # --- EXHAUSTED ALL ATTEMPTS ---
            trace_log("exhausted", trace_id, query, query, attempt_index,
                      best_id=best_result_so_far.get('id') if best_result_so_far else None,
                      best_confidence=best_result_so_far.get('confidence_score', 0.0) if best_result_so_far else 0.0,
                      total_attempts=attempt_index,
                      stop_condition="max_loops_reached" if attempt_index >= config.MAX_PIPELINE_LOOPS else "no_more_queries")

            return best_result_so_far, best_candidates_so_far
        
        # Execute with or without semaphore
        if semaphore:
            async with semaphore:
                return await _run_with_semaphore()
        else:
            return await _run_with_semaphore()

    def _parse_suggestions(self, confidence_result: Optional[Dict]) -> List[str]:
        """Parse suggested alternatives from scorer response."""
        if not confidence_result:
            return []
        
        raw = confidence_result.get('suggested_alternatives') or []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                raw = parsed if isinstance(parsed, list) else [raw]
            except Exception:
                raw = [s.strip() for s in raw.strip('[]').split(',') if s.strip()]
        return [s for s in raw if isinstance(s, str) and s.strip()]

    async def _try_synonym_generation(
        self, query: str, context: Optional[str], feedback: str,
        queries_tried: set, queries_to_try: list, trace_id: str, attempt_index: int
    ) -> None:
        """Attempt synonym generation and add new queries."""
        try:
            trace_log("synonym_generation_start", trace_id, query, query, attempt_index,
                      feedback=feedback[:200])
            
            syns = await self.synonym_generator.generate_synonyms(
                query=query,
                context=context or "",
                feedback=feedback,
                trace_id=trace_id,
            )
            
            if syns:
                already = set(queries_tried) | set(queries_to_try)
                syns = [s for s in syns if s and s not in already and s != query]
                if syns:
                    trace_log("synonym_generation_complete", trace_id, query, query, attempt_index,
                              synonyms=syns)
                    queries_to_try.extend(syns)
                else:
                    trace_log("synonym_generation_complete", trace_id, query, query, attempt_index,
                              synonyms=[], message="All synonyms already tried")
            else:
                trace_log("synonym_generation_complete", trace_id, query, query, attempt_index,
                          synonyms=[], message="No synonyms generated")
        except Exception as e:
            logger.error("Synonym generator failed: %s", e, exc_info=True)
            trace_log("synonym_generation_error", trace_id, query, query, attempt_index,
                      error=str(e))

    def close(self):
        """Closes any open resources, like database connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("Pipeline resources closed.")


def create_pipeline(provider: str = "gemini") -> RAGPipeline:
    """Creates/Instantiates the pipeline with Gemini components."""
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
