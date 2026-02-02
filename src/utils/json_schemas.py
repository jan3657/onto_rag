# src/utils/json_schemas.py
"""
JSON schemas for structured output (guided decoding) with vLLM.

These schemas force the model to output valid JSON matching the expected format,
eliminating parsing errors and thinking block issues.
"""

# Schema for Selector - choose the best ontology term
SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_id": {
            "type": "string",
            "description": "The full ID of the chosen term (e.g., 'CHEBI:15377') or '-1' if no match"
        },
        "explanation": {
            "type": "string",
            "description": "Brief explanation for the selection"
        }
    },
    "required": ["chosen_id", "explanation"]
}

# Schema for Confidence Scorer - assess match quality
SCORER_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score between 0.0 and 1.0"
        },
        "explanation": {
            "type": "string",
            "description": "Justification for the confidence score"
        },
        "suggested_alternatives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative search terms if confidence is low"
        }
    },
    "required": ["confidence_score", "explanation", "suggested_alternatives"]
}

# Schema for Synonym Generator - generate alternative queries
SYNONYM_SCHEMA = {
    "type": "object",
    "properties": {
        "synonyms": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
            "description": "List of up to 5 alternative search queries"
        }
    },
    "required": ["synonyms"]
}
