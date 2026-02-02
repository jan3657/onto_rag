"""
vLLM-compatible LLM client using OpenAI SDK.

This client connects to a vLLM server running with the OpenAI-compatible API,
allowing seamless use of locally-served models in the evaluation pipeline.
"""
from typing import Optional, Tuple, Dict
import asyncio
import logging

from openai import OpenAI, AsyncOpenAI
from src.interfaces import LLMClient

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """
    LLM client for vLLM servers using OpenAI-compatible API.
    
    Usage:
        client = VLLMClient(base_url="http://127.0.0.1:8000/v1")
        response, tokens = await client.generate_json("prompt", model=client.model)
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (e.g., "http://127.0.0.1:8000/v1")
            api_key: API key (vLLM requires one but ignores it, use "EMPTY")
            model: Model name. If None, auto-discovers from server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        
        # Sync client for model discovery
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)
        # Async client for actual generation
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        # Discover model if not provided
        if model:
            self.model = model
        else:
            self.model = self._discover_model()
        
        logger.info(f"VLLMClient initialized with model: {self.model} at {base_url}")
    
    def _discover_model(self) -> str:
        """Auto-discover the model name from the vLLM server."""
        try:
            models = self._sync_client.models.list()
            if models.data:
                model_id = models.data[0].id
                logger.info(f"Auto-discovered model: {model_id}")
                return model_id
            raise ValueError("No models available on vLLM server")
        except Exception as e:
            logger.error(f"Failed to discover model from vLLM server: {e}")
            raise
    
    async def generate_json(
        self,
        prompt: str,
        *,
        model: str,
        timeout_seconds: float = 90.0,
        json_schema: dict = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Generate a response from the vLLM model.
        
        Args:
            prompt: The prompt to send
            model: Model name to use (can differ from self.model)
            timeout_seconds: Maximum time for the request
            json_schema: Optional JSON schema for structured output (guided decoding)
            
        Returns:
            Tuple of (response_text, token_usage_dict) or (None, None) on failure
        """
        attempts = 3
        backoff = 1.0
        last_err: Optional[Exception] = None
        use_structured = json_schema is not None
        
        for attempt in range(attempts):
            try:
                # Use the model parameter, falling back to self.model
                model_to_use = model or self.model
                
                # Build request kwargs
                request_kwargs = {
                    "model": model_to_use,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,  # Deterministic for evaluation
                    "max_tokens": 1024,
                }
                
                # Add structured output if schema provided AND not disabled by fallback
                if json_schema and use_structured:
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": json_schema
                        }
                    }
                
                response = await asyncio.wait_for(
                    self._async_client.chat.completions.create(**request_kwargs),
                    timeout=timeout_seconds
                )
                
                # Extract text
                text = None
                if response.choices and response.choices[0].message:
                    text = response.choices[0].message.content
                
                # Extract token usage
                tokens = None
                if response.usage:
                    tokens = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                    }
                
                return text, tokens
                
            except asyncio.TimeoutError:
                last_err = TimeoutError(f"Request timed out after {timeout_seconds}s")
                logger.warning(f"vLLM request timeout (attempt {attempt + 1}/{attempts})")
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                
                # Check for 500 errors that may be caused by structured output
                if "500" in str(e) or "enginecore" in err_str or "internal" in err_str:
                    if use_structured:
                        logger.warning(
                            f"vLLM 500 error with structured output, retrying without "
                            f"(attempt {attempt + 1}/{attempts}): {e}"
                        )
                        use_structured = False  # Disable for remaining attempts
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                
                logger.warning(f"vLLM request failed (attempt {attempt + 1}/{attempts}): {e}")
                
                # Check if retryable
                if "rate" in err_str or "timeout" in err_str:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                # Non-retryable error
                break
        
        if last_err:
            logger.error(f"vLLM request failed after {attempts} attempts: {last_err}")
            raise last_err
        
        return None, None
    
    async def generate_json_with_timeout(
        self,
        prompt: str,
        *,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Wrapper with explicit timeout (same as generate_json for this client)."""
        return await self.generate_json(
            prompt,
            model=model,
            timeout_seconds=timeout_seconds,
        )
