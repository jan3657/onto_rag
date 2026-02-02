# src/utils/tracing.py
"""
Structured tracing utilities for end-to-end pipeline observability.

Provides:
- trace_id generation (UUID4-based)
- JSON-lines structured logging with consistent schema
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Dedicated logger for trace events - outputs JSON lines
trace_logger = logging.getLogger("pipeline.trace")


def generate_trace_id() -> str:
    """Generate a unique trace ID for a pipeline run."""
    return str(uuid.uuid4())[:12]  # Short but unique enough


def trace_log(
    stage: str,
    trace_id: str,
    query_original: str,
    query_current: Optional[str] = None,
    attempt_index: int = 0,
    **extra_data: Any
) -> None:
    """
    Log a structured trace event as a JSON line.
    
    Args:
        stage: Pipeline stage name (e.g., "retrieval_start", "selection_complete")
        trace_id: Unique identifier for this query's processing
        query_original: The original input query
        query_current: Current query (may differ after rewrites)
        attempt_index: Which retry attempt (0 = first try)
        **extra_data: Additional stage-specific data
    """
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "trace_id": trace_id,
        "query_original": query_original,
        "query_current": query_current or query_original,
        "attempt_index": attempt_index,
    }
    
    # Merge extra data, handling non-serializable values
    for key, value in extra_data.items():
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            event[key] = value
        except (TypeError, ValueError):
            event[key] = str(value)
    
    # Output as single JSON line
    trace_logger.info(json.dumps(event, ensure_ascii=False))


# Score semantics documentation for retrieval
SCORE_SEMANTICS = {
    "lexical": {
        "metric": "BM25",
        "higher_is_better": True,
        "description": "Whoosh BM25 relevance score"
    },
    "vector": {
        "metric": "L2_distance", 
        "higher_is_better": False,
        "description": "FAISS L2 distance (lower = more similar)"
    }
}


def format_candidate_tuple(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Format a candidate as a full tuple for logging."""
    return {
        "ontology": candidate.get("source_ontology", "unknown"),
        "id": candidate.get("id", ""),
        "label": candidate.get("label", ""),
        "score": candidate.get("score", 0.0),
        "source": candidate.get("source", "unknown"),
    }
