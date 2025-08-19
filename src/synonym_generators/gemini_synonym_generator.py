# src/synonym_generators/gemini_synonym_generator.py
import logging
from typing import Optional, Tuple, Dict

from google import genai
from google.api_core import exceptions
from google.genai.types import HttpOptions

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

        self.client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=HttpOptions(timeout=60 * 1000))

    async def _call_llm(self, prompt: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, int]]]]:
        logger.info(f"Sending synonym generation request to Gemini...")
        try:
            generation_config = {'temperature': 0.5, 'max_output_tokens': 128}
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )

            feedback = getattr(response, 'prompt_feedback', None)
            if feedback and any(r.blocked for r in feedback.safety_ratings or []):
                logger.warning(f"Synonym generation request was blocked by safety filters.")
                return None, None
            
            finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            if finish_reason and finish_reason.name == 'MAX_TOKENS':
                logger.warning(
                    f"Gemini response was truncated due to max_output_tokens limit ({generation_config['max_output_tokens']}). "
                    "The output may be incomplete or invalid JSON."
                )

            # Extract token usage if available
            token_usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0)
                }

            return response.text, token_usage
        except exceptions.GoogleAPIError as e:
            logger.error(f"A Google API error occurred with the Gemini call for synonym generation: {e}", exc_info=True)
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred with the Gemini synonym API call: {e}", exc_info=True)
            return None, None