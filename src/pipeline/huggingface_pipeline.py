# src/pipeline/huggingface_pipeline.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.pipeline.base_pipeline import BaseRAGPipeline
from src.rag_selectors.huggingface_selector import HuggingFaceSelector
from src.confidence_scorers.huggingface_confidence_scorer import HuggingFaceConfidenceScorer

class HuggingFaceRAGPipeline(BaseRAGPipeline):
    """
    A RAG pipeline implementation that uses a local Hugging Face model.
    """
    def __init__(self):
        super().__init__(
            selector_class=HuggingFaceSelector, 
            confidence_scorer_class=HuggingFaceConfidenceScorer
        )