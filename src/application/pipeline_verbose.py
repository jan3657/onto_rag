import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src import config
from src.application.pipeline import BaseRAGPipeline

logger = logging.getLogger(__name__)

async def run_pipeline_verbose(
    pipeline: BaseRAGPipeline,
    query: str,
    context: Optional[str] = None,
    lexical_k: int = config.DEFAULT_K_LEXICAL,
    vector_k: int = config.DEFAULT_K_VECTOR,
    target_ontologies: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the pipeline and return rich debugging information for each step.

    Returns
    -------
    Tuple containing:
        - ranked results (list, possibly empty)
        - candidates list corresponding to best result
        - history list with per-iteration details
    """
    queries_to_try = [query]
    queries_tried = set()
    best_results_so_far: List[Dict[str, Any]] = []
    best_candidates_so_far: List[Dict[str, Any]] = []
    loop_count = 0
    last_feedback = ""
    history: List[Dict[str, Any]] = []

    while queries_to_try and loop_count < config.MAX_PIPELINE_LOOPS:
        current_query = queries_to_try.pop(0)
        if current_query in queries_tried:
            continue
        loop_count += 1
        queries_tried.add(current_query)

        step: Dict[str, Any] = {"query": current_query}
        retriever_output = pipeline.retriever.search(
            current_query,
            lexical_limit=lexical_k,
            vector_k=vector_k,
            target_ontologies=target_ontologies,
        )
        candidates = retriever_output.get("lexical_results", []) + retriever_output.get("vector_results", [])
        step["retrieved_entities"] = candidates
        if not candidates:
            history.append(step)
            continue

        selection = await pipeline.selector.select_best_term(
            current_query,
            candidates,
            context=context or "",
            feedback=last_feedback,
        )
        step["selector_prompt"] = getattr(pipeline.selector, "last_prompt", "")
        step["selector_raw_response"] = getattr(pipeline.selector, "last_raw_response", "")
        step["selection"] = selection

        choices = selection.get("choices") if selection else []
        step["choices"] = choices

        scored_results: List[Dict[str, Any]] = []
        scorer_suggestions: List[str] = []

        for choice in choices[:3]:
            chosen_id = choice.get("id")
            chosen_term_details = pipeline.retriever.get_term_details(chosen_id)
            if not chosen_term_details:
                continue
            try:
                confidence_result = await pipeline.confidence_scorer.score_confidence(
                    query=query,
                    chosen_term_details=chosen_term_details,
                    all_candidates=candidates,
                    context=context or "",
                )
            except Exception as e:
                logger.error("Confidence scorer failed: %s", e, exc_info=True)
                confidence_result = None

            step.setdefault("scorer_prompt", getattr(pipeline.confidence_scorer, "last_prompt", ""))
            step.setdefault("scorer_raw_response", getattr(pipeline.confidence_scorer, "last_raw_response", ""))

            current_result = dict(chosen_term_details)
            current_result["selector_explanation"] = choice.get("selector_explanation", "No explanation available.")
            current_result["selector_confidence"] = choice.get("selector_confidence", None)

            if confidence_result is not None:
                score = confidence_result.get("confidence_score")
                current_result["confidence_score"] = score
                current_result["scorer_explanation"] = confidence_result.get("scorer_explanation", "No explanation available.")
                current_result["suggested_alternatives"] = confidence_result.get("suggested_alternatives", [])

                raw = confidence_result.get("suggested_alternatives") or []
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        raw = parsed if isinstance(parsed, list) else [raw]
                    except Exception:
                        raw = [s.strip() for s in raw.strip("[]").split(",") if s.strip()]
                scorer_suggestions.extend([s for s in raw if isinstance(s, str) and s.strip()])
            else:
                current_result["confidence_score"] = -1.0
                current_result["scorer_explanation"] = "Scorer failed to provide explanation."
                current_result["suggested_alternatives"] = []

            scored_results.append(current_result)

        if scored_results:
            scored_results.sort(key=lambda r: r.get("confidence_score", -1), reverse=True)
            step["results"] = scored_results
            step["result"] = scored_results[0]  # backward-compatible single best
            last_feedback = scored_results[0].get("scorer_explanation", "")

            best_score_so_far = best_results_so_far[0].get("confidence_score", -1) if best_results_so_far else -1
            if scored_results[0].get("confidence_score", -1) > best_score_so_far:
                best_results_so_far = scored_results
                best_candidates_so_far = candidates

            if scorer_suggestions:
                step["scorer_suggestions"] = scorer_suggestions
                already = set(queries_tried) | set(queries_to_try)
                to_add = [s for s in scorer_suggestions if s not in already and s != query]
                if to_add:
                    queries_to_try = to_add + queries_to_try

            if scored_results[0].get("confidence_score", 0.0) >= config.CONFIDENCE_THRESHOLD:
                history.append(step)
                return best_results_so_far, best_candidates_so_far, history

        if pipeline.synonym_generator and not suggestions and not queries_to_try:
            try:
                syns = await pipeline.synonym_generator.generate_synonyms(
                    query=query,
                    context=context or "",
                    feedback=last_feedback,
                )
            except Exception as e:
                logger.error("Synonym generator failed: %s", e, exc_info=True)
                syns = []
            step["synonym_prompt"] = getattr(pipeline.synonym_generator, "last_prompt", "")
            step["synonym_raw_response"] = getattr(pipeline.synonym_generator, "last_raw_response", "")
            step["synonyms"] = syns
            if syns:
                already = set(queries_tried) | set(queries_to_try)
                syns = [s for s in syns if s not in already and s != query]
                if syns:
                    queries_to_try.extend(syns)

        history.append(step)

    return best_results_so_far, best_candidates_so_far, history
