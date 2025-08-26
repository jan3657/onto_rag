# adapters/selection/gemini_selector.py
import logging
from typing import Optional, Tuple, Dict

from src.application.selection.base_selector import BaseSelector
from src.domain.ports import Retriever
from src.infrastructure.llm.clients.gemini import GeminiClient
from src import config

logger = logging.getLogger(__name__)


class GeminiSelector(BaseSelector):
    """Uses the Google Gemini model to select the best ontology term."""

    def __init__(self, retriever: Retriever):
        model_name = config.GEMINI_SELECTOR_MODEL_NAME
        super().__init__(retriever=retriever, model_name=model_name)
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = GeminiClient(api_key=config.GEMINI_API_KEY)

    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text, usage = await self.client.generate_json(prompt, model=self.model_name)
        return text, usage
