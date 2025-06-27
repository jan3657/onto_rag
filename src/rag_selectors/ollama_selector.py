# src/rag_selectors/ollama_selector.py
import logging
from typing import Optional

import ollama

from src.rag_selectors.base_selector import BaseSelector
from src.retriever.hybrid_retriever import HybridRetriever
from src import config

logger = logging.getLogger(__name__)

class OllamaSelector(BaseSelector):
    """
    Uses a local LLM via Ollama to select the best ontology term.
    """
    def __init__(self, retriever: HybridRetriever):
        """
        Initializes the OllamaSelector.

        Args:
            retriever (HybridRetriever): An initialized retriever instance.
        """
        model_name = config.OLLAMA_SELECTOR_MODEL_NAME
        super().__init__(retriever=retriever, model_name=model_name)
        
        try:
            ollama.ps()
            logger.info("Ollama service is running.")
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc

    def _call_llm(self, prompt: str, query: str) -> Optional[str]:
        """
        Makes the API call to the Ollama service.
        """
        logger.info(f"Sending request to Ollama for query: '{query}' with model '{self.model_name}'")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                format='json'
            )
            return response['message']['content']
                
        except ollama.ResponseError as e:
            logger.error(f"An error occurred with the Ollama API call: {e.status_code} - {e.error}")
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during the Ollama call: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during the Ollama call: {e}", exc_info=True)
            return None