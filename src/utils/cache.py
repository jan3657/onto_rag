# src/utils/cache.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Define a type hint for the cache data structure for clarity
CacheData = Dict[str, Optional[Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]]]

def load_cache(path: Path) -> CacheData:
    """
    Loads the pipeline cache from a JSON file.

    Args:
        path (Path): The path to the cache file.

    Returns:
        A dictionary representing the cache. Returns an empty dict if the file
        doesn't exist or is invalid.
    """
    if not path.exists():
        logger.warning(f"Cache file not found at {path}. Starting with an empty cache.")
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            logger.info(f"Loading existing pipeline cache from: {path}")
            cache_data = json.load(f)
            logger.info(f"Loaded {len(cache_data)} items from cache.")
            return cache_data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Could not read or parse cache file at {path}: {e}. Starting fresh.")
        return {}

def save_cache(path: Path, cache_data: CacheData):
    """
    Saves the pipeline cache to a JSON file.

    Args:
        path (Path): The path to save the cache file to.
        cache_data (Dict): The cache dictionary to save.
    """
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Successfully saved {len(cache_data)} items to cache at: {path}")
    except IOError as e:
        logger.error(f"Failed to save cache to {path}: {e}")