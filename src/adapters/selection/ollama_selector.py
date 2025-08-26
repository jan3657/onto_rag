# adapters/selection/ollama_selector.py
import logging
from typing import Optional, Tuple, Dict

from src.application.selection.base_selector import BaseSelector
from src.domain.ports import Retriever
from src.infrastructure.llm.clients.ollama import OllamaClient
from src import config

logger = logging.getLogger(__name__)


class OllamaSelector(BaseSelector):
    """Uses a local LLM via Ollama to select the best ontology term."""

    def __init__(self, retriever: Retriever):
        model_name = config.OLLAMA_SELECTOR_MODEL_NAME
        super().__init__(retriever=retriever, model_name=model_name)
        self.client = OllamaClient()

    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
