# adapters/pipeline_factory.py
from src.application.pipeline import BaseRAGPipeline
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.adapters.selection.gemini_selector import GeminiSelector
from src.adapters.selection.ollama_selector import OllamaSelector
from src.adapters.selection.huggingface_selector import HuggingFaceSelector
from src.adapters.confidence.gemini_confidence_scorer import GeminiConfidenceScorer
from src.adapters.confidence.ollama_confidence_scorer import OllamaConfidenceScorer
from src.adapters.confidence.huggingface_confidence_scorer import HuggingFaceConfidenceScorer
from src.adapters.synonyms.gemini_synonym_generator import GeminiSynonymGenerator
from src.adapters.synonyms.ollama_synonym_generator import OllamaSynonymGenerator
from src.adapters.synonyms.huggingface_synonym_generator import HuggingFaceSynonymGenerator


def create_pipeline(provider: str) -> BaseRAGPipeline:
    retriever = HybridRetriever()
    if provider == "gemini":
        selector = GeminiSelector(retriever)
        confidence = GeminiConfidenceScorer()
        synonym = GeminiSynonymGenerator()
    elif provider == "ollama":
        selector = OllamaSelector(retriever)
        confidence = OllamaConfidenceScorer()
        synonym = OllamaSynonymGenerator()
    elif provider == "huggingface":
        selector = HuggingFaceSelector(retriever)
        confidence = HuggingFaceConfidenceScorer()
        synonym = HuggingFaceSynonymGenerator()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return BaseRAGPipeline(
        retriever=retriever,
        selector=selector,
        confidence_scorer=confidence,
        synonym_generator=synonym,
    )

