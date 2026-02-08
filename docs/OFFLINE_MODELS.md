# Offline Model Usage for HPC Compute Nodes

## Problem
Compute nodes don't have internet access, so they can't download HuggingFace models directly.

## Solution

### Step 1: Download Models on Login Node

Run this **on a LOGIN NODE** (with internet access):

```bash
python scripts/download_models.py
```

This will download both models to `models/sentence-transformers/`:
- `all-MiniLM-L6-v2` (~90MB)
- `cambridgeltl_SapBERT-from-PubMedBERT-fulltext` (~420MB)

### Step 2: Run Ingestion on Compute Node

The models are now cached locally. Run ingestion normally:

```bash
# This will use the cached models
python -m src.evaluation.evaluate_craft_chebi --ingest
```

## How It Works

The `MODEL_CACHE_DIR` in `config.py` points to a local directory:
```python
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "sentence-transformers"
```

When `SentenceTransformer` loads a model:
```python
model = SentenceTransformer(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    cache_folder=str(MODEL_CACHE_DIR),
    trust_remote_code=True
)
```

It will:
1. **First** check the cache directory
2. **Only if not found**, try to download from HuggingFace

## Troubleshooting

**If download fails:**
- Make sure you're on a login node with internet access
- Check you have ~500MB free space
- Verify `models/sentence-transformers/` directory exists

**If compute node still tries to download:**
- Verify the cache contains: `models/sentence-transformers/cambridgeltl_SapBERT-from-PubMedBERT-fulltext/`
- Check file permissions (should be readable by your user)

## Manual Download Alternative

If the script doesn't work, you can manually download:

```bash
# On login node
cd models/sentence-transformers/
git lfs install
git clone https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext cambridgeltl_SapBERT-from-PubMedBERT-fulltext
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 all-MiniLM-L6-v2
```
