import json
import os
import logging
import time
from typing import List, Dict, Any

# Ensure src is in path for imports if run directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # To import from src

try:
    from sentence_transformers import SentenceTransformer
    import torch 
except ModuleNotFoundError:
    logging.error("Modules sentence_transformers or torch not found. Please install them: pip install sentence-transformers torch")
    sys.exit(1)

try:
    from src import config # Use 'from src import config'
except ModuleNotFoundError:
    # Fallback for running script directly from src/embeddings where src might not be seen as a package root
    # This is less ideal but can help during direct script execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import config
    except ModuleNotFoundError:
        logging.error("Error: Could not import 'config'. "
                    "Ensure the script is run from the project root or 'src' is in PYTHONPATH.")
        sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    model_name: str, 
    batch_size: int = 32,
    device: str = None
) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of documents using a SentenceTransformer model.

    Args:
        documents: A list of dictionaries, each with at least "id", "label", and "text" keys.
        model_name: The name of the SentenceTransformer model to use.
        batch_size: The number of documents to process in each batch.
        device: The device to use for computation (e.g., "cuda", "cpu"). Auto-detects if None.

    Returns:
        A list of dictionaries, each containing "id", "label", and "embedding" (list of floats).
    """
    if not documents:
        logging.warning("No documents provided for embedding.")
        return []

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info(f"Loading SentenceTransformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        # Fallback to CPU if CUDA error during model loading (e.g. out of memory)
        if "cuda" in str(e).lower() and device == "cuda":
            logging.warning("CUDA error during model load. Attempting to load on CPU.")
            device = "cpu"
            model = SentenceTransformer(model_name, device=device)
        else:
            raise

    logging.info("Model loaded. Starting embedding process...")

    texts_to_embed = [doc.get("text", "") for doc in documents] # Get text, default to empty string if missing
    ids = [doc.get("id") for doc in documents]
    labels = [doc.get("label") for doc in documents]

    all_embeddings_data = []
    start_time = time.time()

    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        logging.info(f"Processing batch {i//batch_size + 1}/{(len(texts_to_embed) -1)//batch_size + 1} (size: {len(batch_texts)})")
        
        try:
            embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False) # Returns numpy array
            
            for idx, embedding in enumerate(embeddings):
                all_embeddings_data.append({
                    "id": batch_ids[idx],
                    "label": batch_labels[idx],
                    "embedding": embedding.tolist() # Convert numpy array to list for JSON serialization
                })
        except Exception as e:
            logging.error(f"Error embedding batch starting at index {i}: {e}")
            # Optionally, decide how to handle batch errors (e.g., skip batch, add placeholders)
            # For now, we'll just log and continue, resulting in missing embeddings for that batch.
            # To add placeholders:
            # for j in range(len(batch_texts)):
            #     all_embeddings_data.append({
            #         "id": batch_ids[j],
            #         "label": batch_labels[j],
            #         "embedding": None # Or a zero vector of correct dimensionality
            #     })


    end_time = time.time()
    logging.info(f"Embedding process completed for {len(all_embeddings_data)} documents in {end_time - start_time:.2f} seconds.")
    
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'model_max_length'):
        logging.info(f"Model max sequence length: {model.tokenizer.model_max_length}")
    elif hasattr(model, 'max_seq_length'):
         logging.info(f"Model max sequence length: {model.max_seq_length}")


    return all_embeddings_data

def save_embeddings(embeddings_data: List[Dict[str, Any]], output_path: str):
    """Saves the embeddings data to a JSON file."""
    if not embeddings_data:
        logging.warning("No embeddings data to save.")
        return

    logging.info(f"Saving {len(embeddings_data)} embeddings to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2) # Use indent for readability, can remove for smaller file size
        logging.info("Successfully saved embeddings.")
    except IOError:
        logging.error(f"Error: Could not write embeddings to {output_path}")
    except TypeError as e:
        logging.error(f"TypeError during JSON serialization: {e}. Check embedding data format.")


def main():
    """Main function to generate and save document embeddings."""
    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    enriched_docs = load_enriched_documents(config.ENRICHED_DOCUMENTS_FILE)
    if not enriched_docs:
        logging.error("Failed to load enriched documents. Exiting.")
        return

    embeddings_data = batch_embed_documents(
        documents=enriched_docs,
        model_name=config.EMBEDDING_MODEL_NAME,
        batch_size=64 # Adjust batch size based on available VRAM/RAM
    )

    if embeddings_data:
        save_embeddings(embeddings_data, config.EMBEDDINGS_FILE)
        logging.info(f"Generated and saved {len(embeddings_data)} embeddings.")
        if embeddings_data:
            sample_embedding = embeddings_data[0]['embedding']
            logging.info(f"Sample embedding vector dimension: {len(sample_embedding) if sample_embedding else 'N/A'}")
            logging.info(f"Sample embedding data point: {json.dumps(embeddings_data[0], indent=2, default=lambda x: str(x)[:100])}") # Truncate long embedding
    else:
        logging.warning("No embeddings were generated.")

if __name__ == "__main__":
    main()