# src/synonym_generators/huggingface_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from src.synonym_generators.base_synonym_generator import BaseSynonymGenerator
from src.rag_selectors.huggingface_generator import HuggingFaceLocalGenerator
from src import config

logger = logging.getLogger(__name__)

class HuggingFaceSynonymGenerator(BaseSynonymGenerator):
    """Uses a local Hugging Face model to generate synonyms."""

    def __init__(self):
        model_name = config.HF_SELECTOR_MODEL_ID # Reuse the same model
        super().__init__(model_name=model_name)
        self.generator = HuggingFaceLocalGenerator()

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        logger.info(f"Sending synonym generation request to local HF model...")
        # A bit of creativity is good for synonyms
        response_text = self.generator.generate(prompt, generation_kwargs={'temperature': 0.4})
        # Local models don't have token cost, so we return None for usage.
        return response_text, None