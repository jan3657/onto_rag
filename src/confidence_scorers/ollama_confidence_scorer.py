# src/confidence_scorers/ollama_confidence_scorer.py
import logging
from typing import Optional, Tuple, Dict

from ollama import AsyncClient, ResponseError

from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src import config

logger = logging.getLogger(__name__)

class OllamaConfidenceScorer(BaseConfidenceScorer):
    """Uses a local LLM via Ollama to assess the confidence of an ontology mapping."""
    
    def __init__(self):
        model_name = config.OLLAMA_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)
        
        self.client = AsyncClient()

        try:
            import ollama
            ollama.ps()
            logger.info("Ollama service is running for confidence scorer.")
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc

    async def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        logger.info(f"Sending confidence scoring request to Ollama with model '{self.model_name}'")
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
                options={'temperature': 0.0}
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
            logger.error(f"An error occurred with the Ollama API call for confidence scoring: {e.status_code} - {e.error}")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during the Ollama confidence scoring call: {e}", exc_info=True)
            return None, None