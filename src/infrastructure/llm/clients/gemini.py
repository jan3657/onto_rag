from typing import Optional, Tuple, Dict
from google import genai
from google.genai.types import HttpOptions
from src.domain.ports import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, timeout_ms: int = 60000):
        self.client = genai.Client(api_key=api_key, http_options=HttpOptions(timeout=timeout_ms))

    async def generate_json(self, prompt: str, *, model: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        resp = await self.client.aio.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
        if text is None and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = getattr(parts[0], "text", None) if parts else None
        usage = getattr(resp, "usage_metadata", None)
        tokens = None if not usage else {
            "prompt_tokens": getattr(usage, "prompt_token_count", 0),
            "completion_tokens": getattr(usage, "candidates_token_count", 0),
        }
        return text, tokens
