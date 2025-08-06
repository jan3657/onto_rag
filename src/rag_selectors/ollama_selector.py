# src/rag_selectors/ollama_selector.py
import logging
from typing import Optional, Tuple, Dict

# CORRECTED: Import AsyncClient
from ollama import AsyncClient, ResponseError

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
        """
        model_name = config.OLLAMA_SELECTOR_MODEL_NAME
        super().__init__(retriever=retriever, model_name=model_name)
        
        self.client = AsyncClient()
        
        try:
            import ollama
            ollama.ps()
            logger.info("Ollama service is running.")
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc

    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Makes the asynchronous API call to the Ollama service using AsyncClient.
        """
        logger.info(f"Sending request to Ollama for query: '{query}' with model '{self.model_name}'")
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                format='json',
                options={
                    'temperature': 0.0
                }
            )
            
            # Extract token usage if available
            token_usage = None
            if 'usage' in response:
                usage = response['usage']
                token_usage = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0)
                }
            
            return response['message']['content'], token_usage
                
        except ResponseError as e:
            logger.error(f"An error occurred with the Ollama API call: {e.status_code} - {e.error}")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during the Ollama call: {e}", exc_info=True)
            return None, None