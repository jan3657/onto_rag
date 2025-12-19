import logging
import json
from typing import List

from src.components.llm_client import GeminiClient
from src import config
from src.utils.token_tracker import token_tracker

logger = logging.getLogger(__name__)

class SynonymGenerator:
    """Uses Google Gemini to generate synonyms for a query."""

    def __init__(self):
        self.model_name = config.GEMINI_SYNONYM_MODEL_NAME
        
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = GeminiClient(api_key=config.GEMINI_API_KEY)
        
        self.prompt_template = self._load_prompt_template()
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        logger.info(f"SynonymGenerator initialized for model: {self.model_name}")

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

    async def generate_synonyms(self, query: str, context: str = "", feedback: str = "") -> List[str]:
        """Formats the prompt, calls the LLM, and parses the response to get synonyms."""
        prompt = (
            self.prompt_template
                .replace("[USER_ENTITY]", query)
                .replace("[CONTEXT]", context or "")
                .replace("[SCORER_FEEDBACK]", feedback or "")
        )

        logger.debug(f"Synonym Generator Prompt:\n---\n{prompt}\n---")
        self.last_prompt = prompt

        response_text, token_usage = await self.client.generate_json(prompt, model=self.model_name)
        self.last_raw_response = response_text or ""
        
        if token_usage:
            token_tracker.record_usage(
                model_name=self.model_name,
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0),
                call_type='synonym_generator'
            )

        if response_text is None:
            return []
            
        logger.debug(f"Synonym Generator Raw Response:\n---\n{response_text}\n---")

        return self._parse_response(response_text)
