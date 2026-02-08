"""
Text normalization helpers used by retrieval and ingestion.
"""

import re
import unicodedata

_SEPARATOR_RE = re.compile(r"[-_/]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_biomedical_text(text: str) -> str:
    """
    Normalize short biomedical text for more stable tokenization.

    This is intentionally conservative:
    - Unicode normalize (NFKC)
    - Replace common symbol separators with spaces
    - Lowercase
    - Collapse whitespace
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = _SEPARATOR_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized
