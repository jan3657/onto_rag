# src/confidence_scorers/base_confidence_scorer.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json

from src import config

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
            
            if 'confidence_score' not in result or 'explanation' not in result:
                logger.error("Confidence scorer response missing required keys. Response: %s", result)
                return None
            
            return {
                'confidence_score': float(result['confidence_score']),
                'explanation': str(result['explanation'])
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to decode or parse confidence scorer response: {response_text}. Error: {e}")
            return None

    @abstractmethod
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Makes the actual API call to the specific LLM provider."""
        pass
    
    def score_confidence(self, query: str, chosen_term_details: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Formats the prompt, calls the LLM, and parses the response for confidence scoring."""
        chosen_details_str = self._format_term_details(chosen_term_details)
        other_candidates_str = self._format_other_candidates(all_candidates, chosen_term_details.get('id', ''))

        prompt = self.prompt_template.replace("[USER_ENTITY]", query)
        prompt = prompt.replace("[CHOSEN_TERM_DETAILS]", chosen_details_str)
        prompt = prompt.replace("[OTHER_CANDIDATES]", other_candidates_str)

        response_text = self._call_llm(prompt)
        if response_text is None:
            return None
        
        return self._parse_response(response_text)