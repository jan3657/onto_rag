from typing import Optional, Tuple, Dict
from src.domain.ports import LLMClient
from src.llm_local.hf_local_generator import HuggingFaceLocalGenerator


class HFLocalClient(LLMClient):
    def __init__(self):
        self.gen = HuggingFaceLocalGenerator()

    async def generate_json(self, prompt: str, *, model: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        text = self.gen.generate(prompt)
        usage = None
        return text, usage
