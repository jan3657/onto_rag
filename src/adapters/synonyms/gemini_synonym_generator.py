# adapters/synonyms/gemini_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from src.application.synonyms.base_synonym_generator import BaseSynonymGenerator
from src.infrastructure.llm.clients.gemini import GeminiClient
from src import config

logger = logging.getLogger(__name__)


class GeminiSynonymGenerator(BaseSynonymGenerator):
    """Uses Google Gemini to generate synonyms for a query."""

    def __init__(self):
        model_name = config.GEMINI_SYNONYM_MODEL_NAME
        super().__init__(model_name=model_name)
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = GeminiClient(api_key=config.GEMINI_API_KEY)

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
