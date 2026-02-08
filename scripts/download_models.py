#!/usr/bin/env python3
"""
Download HuggingFace models to local cache for offline use on compute nodes.

Run this script on a LOGIN NODE with internet access:
    python scripts/download_models.py

This will download both MiniLM and SapBERT models to your MODEL_CACHE_DIR.
Then on offline compute nodes, run with:
    export HF_LOCAL_FILES_ONLY=1
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from sentence_transformers import SentenceTransformer

def download_model(model_name: str, cache_dir: Path):
    """Download a model to the cache directory."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Cache dir: {cache_dir}")
    print(f"{'='*60}")

    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            trust_remote_code=True
        )
        print(f"✅ Successfully downloaded {model_name}")

        # Get the actual local path
        local_path = cache_dir / model_name.replace("/", "_")
        if local_path.exists():
            print(f"   Local path: {local_path}")

        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("HuggingFace Model Downloader")
    print("="*60)
    print(f"Cache directory: {config.MODEL_CACHE_DIR}")
    print("Tip: set HF_LOCAL_FILES_ONLY=1 on offline compute nodes")

    # Ensure cache directory exists
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download both models
    models = [
        config.EMBEDDING_MODEL_NAME,  # MiniLM
        config.SAPBERT_MODEL_NAME,     # SapBERT
    ]

    results = []
    for model_name in models:
        success = download_model(model_name, config.MODEL_CACHE_DIR)
        results.append((model_name, success))

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for model_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")

    all_success = all(success for _, success in results)
    if all_success:
        print("\n✅ All models downloaded successfully!")
        print(f"\nModels are cached in: {config.MODEL_CACHE_DIR}")
        print("\nYou can now run ingestion on compute nodes without internet access.")
    else:
        print("\n❌ Some models failed to download. Check errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
