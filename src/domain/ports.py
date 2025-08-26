from typing import Protocol, List, Dict, Any, Optional, Tuple


class Retriever(Protocol):
    def search(self, query_string: str, *, lexical_limit: int, vector_k: int,
               target_ontologies: Optional[List[str]] = None) -> Dict[str, Any]: ...
    def get_term_details(self, term_id: str) -> Optional[Dict[str, Any]]: ...


class LLMClient(Protocol):
    async def generate_json(self, prompt: str, *, model: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]: ...
