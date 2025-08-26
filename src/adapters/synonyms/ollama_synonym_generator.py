# adapters/synonyms/ollama_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from src.application.synonyms.base_synonym_generator import BaseSynonymGenerator
from src.infrastructure.llm.clients.ollama import OllamaClient
from src import config

logger = logging.getLogger(__name__)


class OllamaSynonymGenerator(BaseSynonymGenerator):
    """Uses a local LLM via Ollama to generate synonyms for a query."""

    def __init__(self):
        model_name = config.OLLAMA_SYNONYM_MODEL_NAME
        super().__init__(model_name=model_name)
        self.client = OllamaClient()

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
