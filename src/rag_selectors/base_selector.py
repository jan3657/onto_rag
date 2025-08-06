# src/rag_selectors/base_selector.py
import logging
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.retriever.hybrid_retriever import HybridRetriever
from src import config
from src.utils.token_tracker import token_tracker

logger = logging.getLogger(__name__)

class BaseSelector(ABC):
    """
    Abstract base class for LLM-based term selectors.

    This class provides the common framework for loading
    prompts, formatting
    candidate lists, and parsing the final LLM response. Subclasses must
    implement the provider-specific `__init__` setup and the `_call_llm` method.
    """

    def __init__(self, retriever: HybridRetriever, model_name: str):
        """
        Initializes the BaseSelector.

        Args:
            retriever (HybridRetriever): An initialized retriever instance.
            model_name (str): The name of the LLM model to use.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.prompt_template = self._load_prompt_template()
        logger.info(f"{self.__class__.__name__} initialized for model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from the file."""
        template_path = config.SELECTOR_PROMPT_TEMPLATE_PATH
        try:
            with template_path.open('r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found at {template_path}")
            raise

    def _format_candidates_for_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """Formats the list of candidate documents into a string for the prompt."""
        formatted_list = []
        for i, candidate in enumerate(candidates):
            term_id = candidate.get("id")
            if not term_id:
                continue

            details = self.retriever.get_term_details(term_id)
            if not details:
                continue

            label = details.get("label", "N/A")
            definition = details.get("definition", "No definition available.")
            synonyms = "; ".join(details.get("synonyms", [])) or "None"

            formatted_list.append(
                f"{i+1}. ID: {term_id}\n"
                f"   Label: {label}\n"
                f"   Definition: {definition}\n"
                f"   Synonyms: {synonyms}"
            )
        return "\n\n".join(formatted_list)
        
    def _parse_and_validate_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parses the JSON string from the LLM and validates its structure.
        
        Args:
            response_text (str): The raw string content from the LLM, expected to be JSON.

        Returns:
            A validated dictionary or None if parsing or validation fails.
        """
        try:
            # Clean up potential markdown code blocks around the JSON
            cleaned_response = response_text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(cleaned_response)

            # --- Centralized Validation Logic ---
            if "chosen_id" not in result or result.get("chosen_id") is None:
                logger.error(
                    "LLM response is invalid: Missing the mandatory 'chosen_id' key. Response: %s",
                    result
                )
                return None
            
            validated_result = {'chosen_id': str(result['chosen_id'])}

            if 'explanation' in result:
                validated_result['explanation'] = result['explanation']
            else:
                logger.warning("LLM response missing 'explanation' key. Using default value.")
                validated_result['explanation'] = 'No explanation provided.'

            return validated_result
            
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response_text}")
            return None

    @abstractmethod
    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Makes the actual API call to the specific LLM provider.

        This method must be implemented by subclasses. It should handle
        provider-specific API calls, authentication, and error handling.
 
        Args:
            prompt (str): The fully formatted prompt to send to the LLM.
            query (str): The original user query, for logging purposes.

        Returns:
            A tuple containing (response_text, token_usage_dict)
        """
        pass

    async def select_best_term(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the best term by executing the full selection logic.

        Args:
            query (str): The original user query.
            candidates (List[Dict[str, Any]]): The list of candidate documents.

        Returns:
            A dictionary with the selection details, or None on failure.
        """
        if not candidates:
            return None

        candidate_str = self._format_candidates_for_prompt(candidates)
        prompt = self.prompt_template.replace("[USER_ENTITY]", query).replace("[CANDIDATE_LIST]", candidate_str)
        
        logger.debug(f"Selector Prompt:\n---\n{prompt}\n---")

        # Delegate the provider-specific call to the subclass
        response_text, token_usage = await self._call_llm(prompt, query)
        
        if token_usage:
            token_tracker.record_usage(
                model_name=self.model_name,
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0),
                call_type='selector'
            )
        
        if response_text is None:
            return None
            
        logger.debug(f"Selector Raw Response:\n---\n{response_text}\n---") 
        
        # Use the centralized parsing and validation method
        return self._parse_and_validate_response(response_text)