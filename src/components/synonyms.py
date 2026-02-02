import logging
import json
from typing import List

from src.components.llm_client import GeminiClient
from src import config
from src.utils.token_tracker import token_tracker
from src.utils.tracing import trace_log

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

    async def generate_synonyms(self, query: str, context: str = "", feedback: str = "", trace_id: str = "") -> List[str]:
        """Formats the prompt, calls the LLM, and parses the response to get synonyms."""
        prompt = (
            self.prompt_template
                .replace("[USER_ENTITY]", query)
                .replace("[CONTEXT]", context or "")
                .replace("[SCORER_FEEDBACK]", feedback or "")
        )

        logger.debug(f"[SYNONYM_GENERATOR_PROMPT] query='{query}' context='{context[:100]}...' feedback='{feedback[:100]}...' | full_prompt:\n{prompt}")
        self.last_prompt = prompt
        
        if trace_id:
            trace_log("llm_synonym_prompt", trace_id, query, query, 0,
                      prompt_length=len(prompt),
                      context_length=len(context),
                      feedback_length=len(feedback))

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
            
        logger.debug(f"[SYNONYM_GENERATOR_RAW_RESPONSE] query='{query}' | raw_text:\n{response_text}")

        parsed = self._parse_response(response_text)
        
        if trace_id:
            trace_log("llm_synonym_response", trace_id, query, query, 0,
                      synonyms=parsed,
                      synonym_count=len(parsed))
        
        return parsed
