# src/confidence_scorers/gemini_confidence_scorer.py
import json
import logging
from typing import Optional, Tuple, Dict
import typing
import typing_extensions

from google import genai
from google.api_core import exceptions
from google.genai import types
from google.genai.types import HttpOptions, GenerateContentConfig, ThinkingConfig

from src.confidence_scorers.base_confidence_scorer import BaseConfidenceScorer
from src import config

logger = logging.getLogger(__name__)

class ConfidenceAssessment(typing_extensions.TypedDict):
    confidence_score: int
    explanation: str
    suggested_alternatives: list[str]


class GeminiConfidenceScorer(BaseConfidenceScorer):
    """Uses Google Gemini to assess the confidence of an ontology mapping."""
    def __init__(self):
        model_name = config.GEMINI_SCORER_MODEL_NAME
        super().__init__(model_name=model_name)

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=HttpOptions(timeout=60 * 1000)
        )

    async def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Makes a standard asynchronous API call to the Gemini model with robust parsing + retry."""

        async def _once(max_out_tokens: int, thinking_budget: int):
            cfg = GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_out_tokens,
                response_mime_type="application/json",
                response_schema=ConfidenceAssessment,
                # For 2.5 Pro we cannot disable thinking, but we can cap it.
                thinking_config=ThinkingConfig(thinking_budget=thinking_budget)
            )
            return await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg
            )

        logger.info("Sending confidence scoring request to Gemini...")
        try:
            # First attempt: reasonable budgets for Pro
            response = await _once(max_out_tokens=1024, thinking_budget=256)

            # Extract token usage (if available)
            token_usage = None
            if getattr(response, "usage_metadata", None):
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    # Useful to see if thinking ate the budget on Pro:
                    "thinking_tokens": getattr(response.usage_metadata, "thinking_token_count", None),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
                }

            # Safety block?
            feedback = getattr(response, "prompt_feedback", None)
            if feedback and any(getattr(r, "blocked", False) for r in (feedback.safety_ratings or [])):
                logger.warning("Confidence scoring request was blocked by safety filters.")
                return None, token_usage

            # Finish reason (handle enum or string)
            finish_reason = None
            try:
                finish_reason = response.candidates[0].finish_reason
                if hasattr(finish_reason, "name"):
                    finish_reason = finish_reason.name
            except Exception:
                pass

            # Prefer parsed JSON (typed), then .text, then parts
            result_json_str = None
            if getattr(response, "parsed", None):
                try:
                    result_json_str = json.dumps(response.parsed)
                except Exception:
                    result_json_str = None

            if not result_json_str and getattr(response, "text", None):
                result_json_str = response.text

            if not result_json_str:
                # Last resort: scan parts for text
                try:
                    parts = response.candidates[0].content.parts or []
                    for p in parts:
                        if getattr(p, "text", None):
                            result_json_str = p.text
                            break
                except Exception:
                    pass

            # If we clearly hit max tokens or still have empty output, retry once with larger budgets
            if (finish_reason == "MAX_TOKENS") or not result_json_str:
                logger.warning(
                    "Gemini response may be truncated/empty (finish_reason=%s). "
                    "Retrying with higher limits.", finish_reason
                )
                response = await _once(max_out_tokens=2048, thinking_budget=384)

                if getattr(response, "usage_metadata", None):
                    token_usage = {
                        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "thinking_tokens": getattr(response.usage_metadata, "thinking_token_count", None),
                        "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
                    }

                # Parse again after retry
                result_json_str = None
                if getattr(response, "parsed", None):
                    try:
                        result_json_str = json.dumps(response.parsed)
                    except Exception:
                        result_json_str = None
                if not result_json_str and getattr(response, "text", None):
                    result_json_str = response.text
                if not result_json_str:
                    try:
                        parts = response.candidates[0].content.parts or []
                        for p in parts:
                            if getattr(p, "text", None):
                                result_json_str = p.text
                                break
                    except Exception:
                        pass

            # Final warning if still truncated
            try:
                fr = response.candidates[0].finish_reason
                if hasattr(fr, "name"):
                    fr = fr.name
                if fr == "MAX_TOKENS":
                    logger.warning("Gemini response was truncated at the final attempt. Output might be incomplete.")
            except Exception:
                pass

            return result_json_str, token_usage

        except exceptions.GoogleAPIError as e:
            logger.error("Google API error in Gemini confidence call: %s", e, exc_info=True)
            return None, None
        except Exception as e:
            logger.error("Unexpected error in Gemini confidence call: %s", e, exc_info=True)
            return None, None
