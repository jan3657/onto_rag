# src/synonym_generators/ollama_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from ollama import AsyncClient, ResponseError

from src.synonym_generators.base_synonym_generator import BaseSynonymGenerator
from src import config

logger = logging.getLogger(__name__)

class OllamaSynonymGenerator(BaseSynonymGenerator):
    """Uses a local LLM via Ollama to generate synonyms for a query."""

    def __init__(self):
        model_name = config.OLLAMA_SYNONYM_MODEL_NAME
        super().__init__(model_name=model_name)
        
        self.client = AsyncClient()
        
        try:
            import ollama
            ollama.ps()
            logger.info("Ollama service is running for synonym generator.")
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        logger.info(f"Sending synonym generation request to Ollama with model '{self.model_name}'")
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
                options={'temperature': 0.2} 
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
            logger.error(f"An error occurred with the Ollama API call for synonym generation: {e.status_code} - {e.error}")
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during the Ollama synonym generation call: {e}", exc_info=True)
            return None, None