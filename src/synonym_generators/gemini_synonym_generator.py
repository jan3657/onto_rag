# src/synonym_generators/gemini_synonym_generator.py
import logging
from typing import Optional

from google import genai
from google.api_core import exceptions

from src.synonym_generators.base_synonym_generator import BaseSynonymGenerator
from src import config

logger = logging.getLogger(__name__)

class GeminiSynonymGenerator(BaseSynonymGenerator):
    """Uses Google Gemini to generate synonyms for a query."""

    def __init__(self):
        model_name = config.GEMINI_SYNONYM_MODEL_NAME
        super().__init__(model_name=model_name)

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def _call_llm(self, prompt: str) -> Optional[str]:
        logger.info(f"Sending synonym generation request to Gemini...")
        try:
            generation_config = {'temperature': 0.2} # Slight creativity is okay
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )

            feedback = getattr(response, 'prompt_feedback', None)
            if feedback and any(r.blocked for r in feedback.safety_ratings or []):
                logger.warning(f"Synonym generation request was blocked by safety filters.")
                return None

            return response.text
        except exceptions.GoogleAPIError as e:
            logger.error(f"A Google API error occurred with the Gemini call for synonym generation: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred with the Gemini synonym API call: {e}", exc_info=True)
            return None