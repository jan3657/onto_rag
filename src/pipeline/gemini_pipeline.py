# src/pipeline/gemini_pipeline.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.pipeline.base_pipeline import BaseRAGPipeline
from src.rag_selectors.gemini_selector import GeminiSelector
from src.confidence_scorers.gemini_confidence_scorer import GeminiConfidenceScorer # ADDED

class GeminiRAGPipeline(BaseRAGPipeline):
    """
    A RAG pipeline implementation that uses Gemini for both selection and confidence scoring.
    """
    def __init__(self):
        """
        Initializes the Gemini-specific RAG pipeline.
        """
        # Pass the Gemini-specific classes to the base constructor
        super().__init__(
            selector_class=GeminiSelector, 
            confidence_scorer_class=GeminiConfidenceScorer
        )