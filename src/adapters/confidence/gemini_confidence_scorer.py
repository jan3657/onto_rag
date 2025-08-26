# adapters/confidence/gemini_confidence_scorer.py
import logging
from typing import Optional, Tuple, Dict

from src.application.confidence.base_confidence_scorer import BaseConfidenceScorer
from src.infrastructure.llm.clients.gemini import GeminiClient
from src import config

logger = logging.getLogger(__name__)


class GeminiConfidenceScorer(BaseConfidenceScorer):
    """Uses Google Gemini to assess the confidence of an ontology mapping."""

    def __init__(self):
        model_name = config.GEMINI_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = GeminiClient(api_key=config.GEMINI_API_KEY)

    async def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
