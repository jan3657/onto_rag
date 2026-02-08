"""
Helpers for robust sentence-transformer model loading.
"""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)


def _offline_hint(model_name: str) -> str:
    return (
        f"Model '{model_name}' is not available in local cache and network access appears unavailable.\n"
        "Run this on a login node with internet access to pre-download models:\n"
        "  .venv/bin/python scripts/download_models.py\n"
        "Then retry on the compute node."
    )


def load_sentence_transformer_model(
    model_name: str,
    *,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Load a sentence-transformer model from shared cache with optional offline mode.

    Behavior:
    - If `config.HF_LOCAL_FILES_ONLY` is true: load from local cache only.
    - Otherwise try normal loading first, then fallback to local-only with a clear
      error message when offline/missing cache.
    """
    kwargs = {
        "trust_remote_code": True,
        "cache_folder": str(config.MODEL_CACHE_DIR),
    }
    if device:
        kwargs["device"] = device

    if getattr(config, "HF_LOCAL_FILES_ONLY", False):
        logger.info(
            "Loading model '%s' with local_files_only=True from cache '%s'",
            model_name,
            config.MODEL_CACHE_DIR,
        )
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)
        except Exception as exc:
            raise RuntimeError(_offline_hint(model_name)) from exc

    try:
        logger.info("Loading model '%s' (network allowed, cache='%s')", model_name, config.MODEL_CACHE_DIR)
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        logger.warning(
            "Model load failed for '%s' with network-enabled mode (%s). "
            "Retrying local-files-only mode.",
            model_name,
            exc,
        )
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)
        except Exception as local_exc:
            raise RuntimeError(_offline_hint(model_name)) from local_exc
