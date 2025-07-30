# src/pipeline/pipeline_factory.py
from src.pipeline.base_pipeline import BaseRAGPipeline
from src.pipeline.gemini_pipeline import GeminiRAGPipeline
from src.pipeline.ollama_pipeline import OllamaRAGPipeline
from src.pipeline.huggingface_pipeline import HuggingFaceRAGPipeline

def get_pipeline(pipeline_name: str) -> BaseRAGPipeline:
    """
    Factory function to get an instance of a RAG pipeline.
    This is the central place to manage pipeline selection.
    """
    if pipeline_name == "gemini":
        return GeminiRAGPipeline()
    elif pipeline_name == "ollama":
        return OllamaRAGPipeline()
    elif pipeline_name == "huggingface":
        return HuggingFaceRAGPipeline()
    else:
        raise ValueError(f"Unknown pipeline name: '{pipeline_name}'. Valid options are 'gemini', 'ollama', 'huggingface'.")