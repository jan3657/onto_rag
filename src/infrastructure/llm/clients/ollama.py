from typing import Optional, Tuple, Dict
from ollama import AsyncClient
from src.domain.ports import LLMClient


class OllamaClient(LLMClient):
    def __init__(self):
        self.client = AsyncClient()

    async def generate_json(self, prompt: str, *, model: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        resp = await self.client.chat(model=model, format="json", messages=[{"role": "user", "content": prompt}])
        msg = resp.get("message", {})
        text = msg.get("content")
        usage = resp.get("eval_count")
        tokens = {"completion_tokens": usage} if usage is not None else None
        return text, tokens
