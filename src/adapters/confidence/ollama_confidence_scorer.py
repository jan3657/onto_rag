# adapters/confidence/ollama_confidence_scorer.py
import logging
from typing import Optional, Tuple, Dict

from src.application.confidence.base_confidence_scorer import BaseConfidenceScorer
from src.infrastructure.llm.clients.ollama import OllamaClient
from src import config

logger = logging.getLogger(__name__)


class OllamaConfidenceScorer(BaseConfidenceScorer):
    """Uses a local LLM via Ollama to assess the confidence of an ontology mapping."""

    def __init__(self):
        model_name = config.OLLAMA_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)
        self.client = OllamaClient()

    async def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
