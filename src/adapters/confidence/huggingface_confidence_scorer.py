# adapters/confidence/huggingface_confidence_scorer.py
import logging
from typing import Optional, Tuple, Dict

from src.application.confidence.base_confidence_scorer import BaseConfidenceScorer
from src.infrastructure.llm.clients.hf_local import HFLocalClient
from src import config

logger = logging.getLogger(__name__)


class HuggingFaceConfidenceScorer(BaseConfidenceScorer):
    """Uses a local Hugging Face model to assess mapping confidence."""

    def __init__(self):
        model_name = config.HF_SELECTOR_MODEL_ID
        super().__init__(model_name=model_name)
        self.client = HFLocalClient()

    async def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
