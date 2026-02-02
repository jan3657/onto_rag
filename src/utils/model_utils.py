"""
Utility functions for model name handling and file naming.

This module centralizes all model name extraction logic to ensure consistent
naming across logs, caches, and result files.
"""
import re
from src import config


def get_current_model_name() -> str:
    """
    Return the model name based on current PIPELINE setting.
    
    Returns the actual model identifier, not the provider name.
    For example: "gemini-2.5-flash-lite" instead of "gemini"
    """
    provider = config.PIPELINE
    
    if provider == "gemini":
        return config.GEMINI_SELECTOR_MODEL_NAME
    elif provider == "vllm":
        return _extract_vllm_model_name()
    elif provider == "ollama":
        return config.OLLAMA_SELECTOR_MODEL_NAME
    elif provider == "huggingface":
        return config.HF_SELECTOR_MODEL_ID
    return f"unknown-{provider}"


def _extract_vllm_model_name() -> str:
    """
    Extract short name from vLLM model path.
    
    Handles HuggingFace cache paths like:
    .../models--Qwen--Qwen2.5-14B-Instruct/snapshots/abc123
    
    Returns: Short model name (e.g., "Qwen-Qwen2.5-14B-Instruct")
    """
    raw = config.VLLM_SELECTOR_MODEL_NAME or config.VLLM_MODEL_NAME or "vllm-model"
    
    # Handle HF cache paths: models--Org--ModelName
    if "models--" in raw:
        match = re.search(r'models--([^/]+--[^/]+)', raw)
        if match:
            return match.group(1).replace("--", "-")
    
    # Handle standard HF format: Org/ModelName
    if "/" in raw:
        return raw.replace("/", "-")
    
    return raw


def sanitize_for_filename(name: str, max_length: int = 60) -> str:
    """
    Make a string safe for use in filenames.
    
    Args:
        name: String to sanitize
        max_length: Maximum length of resulting string
        
    Returns:
        Sanitized string safe for filenames
    """
    # Replace common path/special characters
    name = name.replace("/", "-").replace(":", "_").replace(" ", "_")
    # Remove any remaining invalid characters
    name = re.sub(r'[\\*?"<>|]', "", name)
    return name[:max_length]


def get_model_file_suffix() -> str:
    """
    Return a sanitized model name suitable for file suffixes.
    
    This is the primary function for naming output files, caches, and logs.
    
    Returns:
        String like "gemini-2.5-flash-lite" or "Qwen-Qwen2.5-14B-Instruct"
    """
    return sanitize_for_filename(get_current_model_name())
