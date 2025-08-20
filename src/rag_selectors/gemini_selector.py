# src/rag_selectors/gemini_selector.py
import asyncio
import logging
from typing import Optional, Tuple, Dict
import typing_extensions

from google import genai
from google.api_core import exceptions
from google.genai.types import HttpOptions

from src.rag_selectors.base_selector import BaseSelector
from src.retriever.hybrid_retriever import HybridRetriever
from src import config

logger = logging.getLogger(__name__)

class SelectionResponse(typing_extensions.TypedDict):
    chosen_id: str
    explanation: str
    confidence: float

class GeminiSelector(BaseSelector):
    """
    Uses the Google Gemini model to select the best ontology term.
    """
    def __init__(self, retriever: HybridRetriever):
        """
        Initializes the GeminiSelector.

        Args:
            retriever (HybridRetriever): An initialized retriever instance.
        """
        model_name = config.GEMINI_SELECTOR_MODEL_NAME
        super().__init__(retriever=retriever, model_name=model_name)
        
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=HttpOptions(timeout=60 * 1000))

    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Makes the API call to the Gemini model.
        """
        logger.info(f"Sending request to Gemini for query: '{query}'")
        try:
            generation_config = {
                'temperature': 0.0,
                'max_output_tokens': 512,
                'response_mime_type': 'application/json',
                'response_schema': SelectionResponse,
            }
            
            response = await asyncio.wait_for(self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
            ), timeout=40)

            feedback = getattr(response, 'prompt_feedback', None)
            if feedback and any(r.blocked for r in feedback.safety_ratings or []):
                logger.warning(f"Request for query '{query}' was blocked by safety filters.")
                return None, None

            # Check if the response was cut short
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

            return response.text.strip(), token_usage
                
        except exceptions.GoogleAPIError as e:
            logger.error(f"A Google API error occurred with the Gemini call: {e}", exc_info=True)
            return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred with the Gemini API call: {e}", exc_info=True)
            return None, None