# src/pipeline/gemini_pipeline.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.pipeline.base_pipeline import BaseRAGPipeline
from src.rag_selectors.gemini_selector import GeminiSelector

class GeminiRAGPipeline(BaseRAGPipeline):
    """
    A RAG pipeline implementation that uses the GeminiSelector for the
    final term selection step.
    """
    def __init__(self):
        """
        Initializes the Gemini-specific RAG pipeline.
        """
        # Pass the GeminiSelector class to the base constructor
        super().__init__(selector_class=GeminiSelector)