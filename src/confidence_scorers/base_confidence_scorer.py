# src/confidence_scorers/base_confidence_scorer.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import json

from src import config
from src.utils.token_tracker import token_tracker

logger = logging.getLogger(__name__)

class BaseConfidenceScorer(ABC):
    """Abstract base class for LLM-based confidence scorers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_template = self._load_prompt_template()
        logger.info(f"{self.__class__.__name__} initialized for model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        template_path = config.CONFIDENCE_PROMPT_TEMPLATE_PATH
        try:
            with template_path.open('r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Confidence prompt template not found at {template_path}")
            raise

    def _format_term_details(self, term_details: Dict[str, Any]) -> str:
        label = term_details.get("label", "N/A")
        definition = term_details.get("definition", "No definition available.")
        synonyms = "; ".join(term_details.get("synonyms", [])) or "None"
        return f"ID: {term_details.get('id', 'N/A')}\nLabel: {label}\nDefinition: {definition}\nSynonyms: {synonyms}"

    def _format_other_candidates(self, candidates: List[Dict[str, Any]], chosen_id: str, limit: int = 5) -> str:
        other_candidates = [c for c in candidates if c.get('id') != chosen_id][:limit]
        if not other_candidates:
            return "None provided."
        
        formatted_list = []
        for cand in other_candidates:
            formatted_list.append(f"- {cand.get('label', 'N/A')} (ID: {cand.get('id', 'N/A')})")
        return "\n".join(formatted_list)
    
    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned_response = response_text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(cleaned_response)

            if 'confidence_score' not in result and 'explanation' not in result:
                logger.error("Confidence scorer response missing required keys. Response: %s", result)
                return {
                    'confidence_score': -1.0,
                    'scorer_explanation': None,
                    'suggested_alternatives': []
                }

            # Normalize suggested_alternatives to a list[str]
            suggestions = result.get('suggested_alternatives', [])
            if isinstance(suggestions, str):
                # try to parse a JSON string; if that fails, split on commas
                try:
                    parsed = json.loads(suggestions)
                    if isinstance(parsed, list):
                        suggestions = parsed
                    else:
                        suggestions = [str(parsed)]
                except Exception:
                    suggestions = [s.strip() for s in suggestions.strip('[]').split(',') if s.strip()]
            elif not isinstance(suggestions, list):
                suggestions = [str(suggestions)]

            # Ensure clean strings
            suggestions = [str(s) for s in suggestions if s is not None]

            return {
                'confidence_score': float(result['confidence_score']),
                'scorer_explanation': str(result['explanation']),
                'suggested_alternatives': suggestions,
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to decode or parse confidence scorer response: {response_text}. Error: {e}")
            return None


    @abstractmethod
    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        """
        Makes the actual API call to the specific LLM provider.

        Returns:
            A tuple containing (response_text, token_usage_dict)
        """
        pass

    async def score_confidence(self, query: str, chosen_term_details: Dict[str, Any], all_candidates: List[Dict[str, Any]], context: str = "") -> Optional[Dict[str, Any]]:
        """Formats the prompt, calls the LLM, and parses the response for confidence scoring."""
        chosen_details_str = self._format_term_details(chosen_term_details)
        other_candidates_str = self._format_other_candidates(all_candidates, chosen_term_details.get('id', ''))

        prompt = (self.prompt_template
                  .replace("[USER_ENTITY]", query)
                  .replace("[CHOSEN_TERM_DETAILS]", chosen_details_str)
                  .replace("[OTHER_CANDIDATES]", other_candidates_str)
                  .replace("[CONTEXT]", context or ""))
        logger.debug(f"Formatted prompt for confidence scoring:\n{prompt}")

        response_text, token_usage = await self._call_llm(prompt)

        if token_usage:
            token_tracker.record_usage(
                model_name=self.model_name,
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0),
                call_type='scorer'
            )

        if response_text is None:
            return None
        
        return self._parse_response(response_text)