# src/embeddings/embed_docs.py
import json
import os
import logging
import time
from typing import List, Dict, Any

from src.utils.logging_config import setup_run_logging

try:
    from sentence_transformers import SentenceTransformer
    import torch 
except ModuleNotFoundError:
    logging.error("Modules sentence_transformers or torch not found. Please install them: pip install sentence-transformers torch")
    sys.exit(1)

# Changed: Use central config and specific constants
from src.config import ONTOLOGIES_CONFIG, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE

setup_run_logging()

def load_enriched_documents(file_path: str) -> List[Dict[str, Any]]:
    """Loads enriched documents from a JSON file."""
    logging.info(f"Loading enriched documents from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logging.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except FileNotFoundError:
        logging.error(f"Error: Enriched documents file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        return []

def batch_embed_documents(
    documents: List[Dict[str, Any]], 
    model: SentenceTransformer,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of documents using a pre-loaded SentenceTransformer model.

    Args:
        documents: A list of dictionaries, each with "id", "label", and "text" keys.
        model: The pre-loaded SentenceTransformer model instance.
        batch_size: The number of documents to process in each batch.

    Returns:
        A list of dictionaries, each containing "id", "label", and "embedding".
    """
    if not documents:
        logging.warning("No documents provided for embedding.")
        return []

    logging.info("Starting embedding process...")
    texts_to_embed = [doc.get("text", "") for doc in documents]
    ids = [doc.get("id") for doc in documents]
    labels = [doc.get("label") for doc in documents]

    all_embeddings_data = []
    start_time = time.time()

    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        logging.info(f"  Processing batch {i//batch_size + 1}/{(len(texts_to_embed) - 1)//batch_size + 1} (size: {len(batch_texts)})")
        
        try:
            embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
            
            for idx, embedding in enumerate(embeddings):
                all_embeddings_data.append({
                    "id": batch_ids[idx],
                    "label": batch_labels[idx],
                    "embedding": embedding.tolist()
                })
        except Exception as e:
            logging.error(f"Error embedding batch starting at index {i}: {e}")

    end_time = time.time()
    logging.info(f"Embedding process completed for {len(all_embeddings_data)} documents in {end_time - start_time:.2f} seconds.")
    return all_embeddings_data

def save_embeddings(embeddings_data: List[Dict[str, Any]], output_path: str):
    """Saves the embeddings data to a JSON file."""
    if not embeddings_data:
        logging.warning("No embeddings data to save.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logging.info(f"Saving {len(embeddings_data)} embeddings to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f) # No indent for smaller file size
        logging.info("Successfully saved embeddings.")
    except IOError as e:
        logging.error(f"Error: Could not write embeddings to {output_path}: {e}")
    except TypeError as e:
        logging.error(f"TypeError during JSON serialization: {e}. Check embedding data format.")

def main():
    """Main function to generate and save document embeddings for all configured ontologies."""
    logging.info("--- Starting Embedding Generation for All Ontologies ---")

    # 1. Load the model once to be reused for all ontologies
    logging.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE, trust_remote_code=True)
        logging.info(f"Model loaded successfully on device: '{EMBEDDING_DEVICE}'")
        if hasattr(model, 'max_seq_length'):
             logging.info(f"Model max sequence length: {model.max_seq_length}")
    except Exception as e:
        logging.error(f"Fatal error loading model: {e}")
        return

    # 2. Loop over all configured ontologies
    for name, config_data in ONTOLOGIES_CONFIG.items():
        enriched_docs_path = config_data.get('enriched_docs_path')
        embeddings_output_path = config_data.get('embeddings_path')
        
        logging.info(f"\n--- Processing Ontology: {name} ---")

        if not enriched_docs_path or not embeddings_output_path:
            logging.warning(f"Config for '{name}' is missing 'enriched_docs_path' or 'embeddings_path'. Skipping.")
            continue
        
        logging.info(f"Input: {enriched_docs_path}")
        logging.info(f"Output: {embeddings_output_path}")

        enriched_docs = load_enriched_documents(enriched_docs_path)
        if not enriched_docs:
            logging.error(f"Failed to load or empty enriched documents file. Skipping '{name}'.")
            continue

        embeddings_data = batch_embed_documents(
            documents=enriched_docs,
            model=model, # Pass the single, pre-loaded model
            batch_size=EMBEDDING_BATCH_SIZE
        )

        if embeddings_data:
            save_embeddings(embeddings_data, embeddings_output_path)
            logging.info(f"Generated and saved {len(embeddings_data)} embeddings for '{name}'.")
        else:
            logging.warning(f"No embeddings were generated for '{name}'.")
    
    logging.info("\n--- All Embedding Generation Complete ---")

if __name__ == "__main__":
    main()