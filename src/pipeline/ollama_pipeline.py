# src/pipeline/ollama_pipeline.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

from src.pipeline.base_pipeline import BaseRAGPipeline
from src.rag_selectors.ollama_selector import OllamaSelector

class OllamaRAGPipeline(BaseRAGPipeline):
    """
    A RAG pipeline implementation that uses the OllamaSelector for the
    final term selection step.
    """
    def __init__(self):
        """
        Initializes the Ollama-specific RAG pipeline.
        """
        # Pass the OllamaSelector class to the base constructor
        super().__init__(selector_class=OllamaSelector)