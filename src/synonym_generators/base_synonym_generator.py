# src/synonym_generators/base_synonym_generator.py
import logging
import json
from abc import ABC, abstractmethod
from typing import List, Optional

from src import config

logger = logging.getLogger(__name__)

class BaseSynonymGenerator(ABC):
    """Abstract base class for LLM-based synonym generators."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_template = self._load_prompt_template()
        logger.info(f"{self.__class__.__name__} initialized for model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        template_path = config.SYNONYM_PROMPT_TEMPLATE_PATH
        try:
            with template_path.open('r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Synonym prompt template not found at {template_path}")
            raise

    def _parse_response(self, response_text: str) -> List[str]:
        try:
            cleaned_response = response_text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(cleaned_response)
            synonyms = result.get("synonyms", [])
            if isinstance(synonyms, list):
                return [s for s in synonyms if isinstance(s, str)]
            logger.warning("Synonym generator response 'synonyms' key is not a list. Response: %s", result)
            return []
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to decode or parse synonym generator response: {response_text}. Error: {e}")
            return []

    @abstractmethod
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Makes the actual API call to the specific LLM provider."""
        pass

    async def generate_synonyms(self, query: str) -> List[str]:
        """Formats the prompt, calls the LLM, and parses the response to get synonyms."""
        prompt = self.prompt_template.replace("[USER_ENTITY]", query)
        
        logger.debug(f"Synonym Generator Prompt:\n---\n{prompt}\n---") # <-- ADD THIS

        response_text = await self._call_llm(prompt)
        if response_text is None:
            return []
            
        logger.debug(f"Synonym Generator Raw Response:\n---\n{response_text}\n---") # <-- ADD THIS

        return self._parse_response(response_text)