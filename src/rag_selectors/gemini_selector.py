# src/rag_selectors/gemini_selector.py
import asyncio
import logging
import random
from typing import Optional, Tuple, Dict
import typing_extensions

from google import genai
from google.api_core import exceptions
from google.genai.types import HttpOptions
from google.genai import errors as genai_errors
from google.genai import types

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
        Returns (response_text, token_usage) or (None, None) on failure.
        Never raises upstream.
        """
        async def _once(max_out_tokens: int):
            cfg = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_out_tokens,
                response_mime_type='application/json',
                response_schema=SelectionResponse,
            )
            return await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg
            )

        # --- retry loop for transient server errors ---
        max_retries = 3
        backoff = 0.5
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await _once(max_out_tokens=512)

                # handle truncation (MAX_TOKENS) by a single enlarge retry
                try:
                    fr = response.candidates[0].finish_reason
                    fr = fr.name if hasattr(fr, "name") else fr
                except Exception:
                    fr = None

                if fr == "MAX_TOKENS":
                    logger.warning("Gemini response may be truncated/empty (finish_reason=MAX_TOKENS) for query %r. Retrying with higher token limit.", query)
                    response = await _once(max_out_tokens=1024)

                # token usage (best effort)
                token_usage = None
                if getattr(response, "usage_metadata", None):
                    token_usage = {
                        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    }

                # prefer .text; fallback to parts
                text = getattr(response, "text", None)
                if not text:
                    try:
                        for p in response.candidates[0].content.parts or []:
                            if getattr(p, "text", None):
                                text = p.text
                                break
                    except Exception:
                        pass

                return text, token_usage

            except genai_errors.ServerError as e:
                last_exc = e
                logger.error("Gemini selector server error (attempt %d/%d): %s", attempt, max_retries, e)
            except Exception as e:
                last_exc = e
                logger.error("Unexpected Gemini selector error (attempt %d/%d): %s", attempt, max_retries, e, exc_info=True)

            # backoff before next attempt
            if attempt < max_retries:
                await asyncio.sleep(backoff + random.random() * 0.25)
                backoff *= 2

        # after retries exhausted
        logger.error("Gemini selector failed after retries: %s", last_exc)
        return None, None