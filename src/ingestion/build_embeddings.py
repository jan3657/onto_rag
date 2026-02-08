# src/ingestion/build_embeddings.py
"""
Generate embeddings for ontology terms and build a FAISS vector index.

Two-step process:
1. build_embeddings() - Generate embeddings JSON from ontology dump
2. build_faiss_index() - Create FAISS index from embeddings JSON
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src import config
from src.utils.model_loading import load_sentence_transformer_model
from src.utils.text_normalization import normalize_biomedical_text

logger = logging.getLogger(__name__)


def build_embeddings(
    ontology_dump_path: Path,
    output_path: Path,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """
    Generate embeddings for ontology terms.

    Creates a JSON file with structure:
    [
        {"id": "CURIE:123", "label": "term name", "embedding": [0.1, 0.2, ...]},
        ...
    ]

    Parameters
    ----------
    ontology_dump_path : Path
        Path to the ontology dump JSON file.
    output_path : Path
        Path where the embeddings JSON will be saved.
    model_name : Optional[str]
        Sentence transformer model to use.
        Default: config.EMBEDDING_MODEL_NAME
    batch_size : Optional[int]
        Batch size for encoding.
        Default: config.EMBEDDING_BATCH_SIZE
    device : Optional[str]
        Device to use ('cpu', 'cuda', 'mps').
        Default: config.EMBEDDING_DEVICE
    """
    # Apply defaults from config
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    device = device or config.EMBEDDING_DEVICE
    model_name_lc = model_name.lower()
    is_sapbert_model = "sapbert" in model_name_lc or "pubmedbert" in model_name_lc

    logger.info(f"Building embeddings from: {ontology_dump_path}")
    logger.info(f"Model: {model_name}, Batch size: {batch_size}, Device: {device}")

    if not ontology_dump_path.exists():
        raise FileNotFoundError(f"Ontology dump not found: {ontology_dump_path}")

    # Load ontology dump
    with ontology_dump_path.open("r", encoding="utf-8") as f:
        ontology_data: Dict[str, Any] = json.load(f)

    logger.info(f"Loaded {len(ontology_data)} terms from dump")

    # Prepare texts for embedding
    # Combine label + synonyms + definition for richer representations
    term_ids: List[str] = []
    term_labels: List[str] = []
    texts_to_embed: List[str] = []

    for curie, term_data in ontology_data.items():
        label = term_data.get("label", "")
        if not label:
            continue

        # Build embedding text: label + synonyms + definition snippet
        parts = [label]
        synonyms = term_data.get("synonyms", [])
        if synonyms:
            parts.append(" ; ".join(synonyms[:5]))  # Limit synonyms
        definition = term_data.get("definition", "")
        if definition:
            parts.append(definition[:200])  # Truncate long definitions

        # SapBERT tokenization is sensitive to uppercase symbol forms
        # (e.g., STAT1/BMP7 can become [UNK]). Add normalized variants
        # so index vectors preserve symbol identity better.
        if is_sapbert_model:
            normalized_label = normalize_biomedical_text(label)
            if normalized_label and normalized_label != label:
                parts.append(normalized_label)

            if synonyms:
                normalized_synonyms = []
                seen_norm = set()
                for syn in synonyms[:5]:
                    norm_syn = normalize_biomedical_text(syn)
                    if not norm_syn or norm_syn in seen_norm or norm_syn == syn:
                        continue
                    normalized_synonyms.append(norm_syn)
                    seen_norm.add(norm_syn)
                if normalized_synonyms:
                    parts.append(" ; ".join(normalized_synonyms))

        embed_text = " | ".join(parts)

        term_ids.append(curie)
        term_labels.append(label)
        texts_to_embed.append(embed_text)

    logger.info(f"Prepared {len(texts_to_embed)} terms for embedding")

    # Load model and generate embeddings
    logger.info(f"Loading embedding model: {model_name}")
    model = load_sentence_transformer_model(
        model_name,
        device=device,
    )

    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    # Build output data
    embeddings_data: List[Dict[str, Any]] = []
    for i, (curie, label) in enumerate(zip(term_ids, term_labels)):
        embeddings_data.append({
            "id": curie,
            "label": label,
            "embedding": embeddings[i].tolist(),
        })

    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, ensure_ascii=False)

    logger.info(f"Saved {len(embeddings_data)} embeddings to: {output_path}")


def build_faiss_index(
    embeddings_path: Path,
    faiss_index_path: Path,
    faiss_metadata_path: Path,
) -> None:
    """
    Build a FAISS index from embeddings JSON.

    Parameters
    ----------
    embeddings_path : Path
        Path to the embeddings JSON file (output of build_embeddings).
    faiss_index_path : Path
        Path where the FAISS index binary will be saved.
    faiss_metadata_path : Path
        Path where the FAISS metadata JSON will be saved.
    """
    import faiss

    logger.info(f"Building FAISS index from: {embeddings_path}")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    # Load embeddings
    with embeddings_path.open("r", encoding="utf-8") as f:
        embeddings_data: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(embeddings_data)} embeddings")

    # Extract vectors and metadata
    vectors: List[List[float]] = []
    metadata: List[Dict[str, str]] = []

    for item in embeddings_data:
        embedding = item.get("embedding")
        if embedding:
            vectors.append(embedding)
            metadata.append({
                "id": item["id"],
                "label": item["label"],
            })

    if not vectors:
        raise ValueError("No valid embeddings found in file")

    # Convert to numpy array
    vectors_np = np.array(vectors, dtype="float32")
    dimension = vectors_np.shape[1]

    logger.info(f"Building FAISS index with {len(vectors)} vectors of dimension {dimension}")

    # Create index (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Save index
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))
    logger.info(f"Saved FAISS index to: {faiss_index_path}")

    # Save metadata
    faiss_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with faiss_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved FAISS metadata ({len(metadata)} items) to: {faiss_metadata_path}")


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.build_embeddings <dump.json> <output.json>")
        print("       python -m src.ingestion.build_embeddings --faiss <embeddings.json> <index.bin> <metadata.json>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    if sys.argv[1] == "--faiss":
        build_faiss_index(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        build_embeddings(Path(sys.argv[1]), Path(sys.argv[2]))
