# src/confidence_scorers/gemini_confidence_scorer.py
import logging
from typing import Optional

from google import genai
from google.api_core import exceptions

from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src import config

logger = logging.getLogger(__name__)

class GeminiConfidenceScorer(BaseConfidenceScorer):
    """Uses Google Gemini to assess the confidence of an ontology mapping."""
    def __init__(self):
        model_name = config.GEMINI_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)
        
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Makes a standard asynchronous API call to the Gemini model.
        """
        logger.info(f"Sending confidence scoring request to Gemini...")
        try:
            generation_config = {
                'temperature': 0.2, 
                'max_output_tokens': 256
            }
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )

            feedback = getattr(response, 'prompt_feedback', None)
            if feedback and any(r.blocked for r in feedback.safety_ratings or []):
                logger.warning("Confidence scoring request was blocked by safety filters.")
                return None

            # Check if the response was cut short
            finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            if finish_reason and finish_reason.name == 'MAX_TOKENS':
                logger.warning(
                    f"Gemini response was truncated due to max_output_tokens limit ({generation_config['max_output_tokens']}). "
                    "The output may be incomplete or invalid JSON."
                )

            return response.text

        except exceptions.GoogleAPIError as e:
            logger.error(f"A Google API error occurred with the Gemini call for confidence scoring: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred with the Gemini confidence API call: {e}", exc_info=True)
            return None