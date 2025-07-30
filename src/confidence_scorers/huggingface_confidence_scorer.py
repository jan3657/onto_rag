# src/confidence_scorers/huggingface_confidence_scorer.py
import logging
from typing import Optional

from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src.rag_selectors.huggingface_generator import HuggingFaceLocalGenerator
from src import config

logger = logging.getLogger(__name__)

class HuggingFaceConfidenceScorer(BaseConfidenceScorer):
    """Uses a local Hugging Face model to assess mapping confidence."""

    def __init__(self):
        model_name = config.HF_SELECTOR_MODEL_ID # Reuse the same model
        super().__init__(model_name=model_name)
        self.generator = HuggingFaceLocalGenerator()

    def _call_llm(self, prompt: str) -> Optional[str]:
        logger.info(f"Sending confidence scoring request to local HF model...")
        return self.generator.generate(prompt, generation_kwargs={'temperature': 0.0})