# src/rag_selectors/huggingface_selector.py
import logging
from typing import Optional, Tuple, Dict

from src.rag_selectors.base_selector import BaseSelector
from src.retriever.hybrid_retriever import HybridRetriever
from src.rag_selectors.huggingface_generator import HuggingFaceLocalGenerator
from src import config

logger = logging.getLogger(__name__)

class HuggingFaceSelector(BaseSelector):
    """
    Uses a local Hugging Face model to select the best ontology term.
    """
    def __init__(self, retriever: HybridRetriever):
        model_name = config.HF_SELECTOR_MODEL_ID
        super().__init__(retriever=retriever, model_name=model_name)
        
        # This will create or get the singleton instance of our generator
        self.generator = HuggingFaceLocalGenerator()

    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Makes the call to the local Hugging Face model.
        """
        logger.info(f"Sending request to local HF model for query: '{query}'")
        
        # The prompt is already formatted by the base class. We just pass it.
        response_text = self.generator.generate(prompt)
        # Local models don't have token cost, so we return None for usage.
        return response_text, None