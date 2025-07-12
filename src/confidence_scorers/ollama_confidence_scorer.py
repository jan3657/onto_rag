# src/rag_selectors/ollama_confidence_scorer.py
import logging
from typing import Optional

import ollama

from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src import config

logger = logging.getLogger(__name__)

class OllamaConfidenceScorer(BaseConfidenceScorer):
    """Uses a local LLM via Ollama to assess the confidence of an ontology mapping."""
    
    def __init__(self):
        # Use a specific model for confidence scoring if needed, or reuse the selector model
        model_name = config.OLLAMA_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)
        
        try:
            ollama.ps()
            logger.info("Ollama service is running for confidence scorer.")
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc

    def _call_llm(self, prompt: str) -> Optional[str]:
        logger.info(f"Sending confidence scoring request to Ollama with model '{self.model_name}'")
        try:
            response = ollama.chat(
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
            return response['message']['content']
        except ollama.ResponseError as e:
            logger.error(f"An error occurred with the Ollama API call for confidence scoring: {e.status_code} - {e.error}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during the Ollama confidence scoring call: {e}", exc_info=True)
            return None