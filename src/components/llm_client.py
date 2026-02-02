from typing import Optional, Tuple, Dict
import asyncio
import random
from google import genai
from google.genai.types import HttpOptions
from google.genai.errors import ClientError, ServerError
from src.interfaces import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, timeout_ms: int = 60000):
        self.client = genai.Client(api_key=api_key, http_options=HttpOptions(timeout=timeout_ms))

    async def generate_json(self, prompt: str, *, model: str, timeout_seconds: float = 90.0) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Call Gemini with light retry/backoff on 429 quota errors.
        
        Args:
            prompt: The prompt to send
            model: Model name to use
            timeout_seconds: Maximum time for entire operation including retries
        """
        attempts = 5
        backoff = 2.0
        last_err: Optional[Exception] = None
        for _ in range(attempts):
            try:
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
            except (ClientError, ServerError) as e:
                last_err = e
                # Retry on quota/rate errors and transient server errors
                msg = str(e)
                status = getattr(e, "status_code", None)
                should_retry = (
                    status in {429, 500, 503, 504}
                    or "RESOURCE_EXHAUSTED" in msg
                    or "DEADLINE_EXCEEDED" in msg
                )
                if not should_retry:
                    raise
                # Backoff with jitter, honor suggested retry if present
                delay = backoff + random.uniform(0, 0.5)
                # Respect explicit retry hints when available
                try:
                    # Some errors include a protobuf RetryInfo with 'retryDelay: 15s'
                    if "retryDelay" in msg:
                        # crude parse to seconds
                        idx = msg.find("retryDelay")
                        seg = msg[idx: idx + 40]
                        import re
                        m = re.search(r"retryDelay\W+(\d+)s", seg)
                        if m:
                            delay = float(m.group(1))
                except Exception:
                    pass
                await asyncio.sleep(min(delay, 30.0))
                backoff = min(backoff * 2, 30.0)
                continue
            except Exception as e:
                last_err = e
                break
        # Exhausted retries
        if last_err:
            raise last_err
        return None, None
    
    async def generate_json_with_timeout(self, prompt: str, *, model: str, timeout_seconds: float = 90.0) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Wrapper that adds asyncio-level timeout protection."""
        try:
            return await asyncio.wait_for(
                self.generate_json(prompt, model=model, timeout_seconds=timeout_seconds),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")
