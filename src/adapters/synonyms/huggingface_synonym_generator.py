# adapters/synonyms/huggingface_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from src.application.synonyms.base_synonym_generator import BaseSynonymGenerator
from src.infrastructure.llm.clients.hf_local import HFLocalClient
from src import config

logger = logging.getLogger(__name__)


class HuggingFaceSynonymGenerator(BaseSynonymGenerator):
    """Uses a local Hugging Face model to generate synonyms."""

    def __init__(self):
        model_name = config.HF_SELECTOR_MODEL_ID
        super().__init__(model_name=model_name)
        self.client = HFLocalClient()

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
