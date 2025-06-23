# src/vector_store/faiss_store.py
import faiss
import json
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, index_path, metadata_path, embeddings_file_path=None, dimension=None):
        """
        Initializes the FAISSVectorStore.
        Tries to load an existing index and metadata. If not found, and an
        embeddings_file_path is provided, it will attempt to build them.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embeddings_file_path = embeddings_file_path
        self.dimension = dimension

        self.index = None
        self.metadata = []

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info(f"Loading existing FAISS index from {self.index_path} and metadata from {self.metadata_path}")
            try:
                self.load_store()
            except Exception as e:
                logger.error(f"Error loading existing FAISS store: {e}. Will attempt to rebuild.", exc_info=True)
                self._try_build_store()
        elif self.embeddings_file_path and os.path.exists(self.embeddings_file_path):
            logger.info(f"FAISS index/metadata not found. Attempting to build from {self.embeddings_file_path}")
            self._try_build_store()
        else:
            message = "FAISS store cannot be loaded or built. "
            if self.embeddings_file_path:
                message += f"Embeddings file '{self.embeddings_file_path}' not found."
            else:
                message += "No embeddings file path provided to build a new store."
            logger.warning(message)

    def _try_build_store(self):
        """Helper method to attempt building the store."""
        embeddings_data = self._load_embeddings_data()
        if embeddings_data:
            self.build_index_from_embeddings(embeddings_data)
            if self.index is not None and self.metadata:
                 self.save_store()
            else:
                logger.warning("FAISS index or metadata not built successfully. Store will not be saved.")
        else:
            logger.warning(f"No embeddings data loaded from {self.embeddings_file_path}, FAISS index not built.")

    def _load_embeddings_data(self):
        """Loads and validates embeddings data from the specified JSON file."""
        if not self.embeddings_file_path or not os.path.exists(self.embeddings_file_path):
            logger.error(f"Embeddings file not found: {self.embeddings_file_path}")
            return None
        try:
            with open(self.embeddings_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list) or not data:
                logger.error(f"Embeddings file {self.embeddings_file_path} is empty or not a list.")
                return None
            if not all('id' in item and 'label' in item and 'embedding' in item for item in data):
                logger.error(f"Embeddings data in {self.embeddings_file_path} has an incorrect format.")
                return None
            logger.info(f"Loaded {len(data)} embeddings from {self.embeddings_file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load embeddings file {self.embeddings_file_path}: {e}", exc_info=True)
            return None

    def build_index_from_embeddings(self, embeddings_data):
        """Builds the FAISS index and prepares metadata from loaded embeddings data."""
        if not embeddings_data:
            logger.warning("No embeddings data provided to build index.")
            return

        try:
            vectors = [item['embedding'] for item in embeddings_data if item.get('embedding')]
            self.metadata = [{'id': item['id'], 'label': item['label']} for item in embeddings_data if item.get('embedding')]
            
            if not vectors:
                logger.warning("No valid vectors found in embeddings_data after filtering.")
                return
                
            vectors_np = np.array(vectors, dtype='float32')

            if not self.dimension:
                self.dimension = vectors_np.shape[1]
            elif self.dimension != vectors_np.shape[1]:
                raise ValueError(f"Provided dimension {self.dimension} does not match embeddings dimension {vectors_np.shape[1]}")

            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_np)
            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors of dimension {self.dimension}.")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.index = None
            self.metadata = []

    def save_store(self):
        """Saves the FAISS index and metadata to their respective files."""
        if self.index is None:
            logger.warning("No FAISS index to save.")
            return

        # *** THIS IS THE RESTORED SAFETY CHECK ***
        if not self.metadata:
            logger.warning("FAISS index exists but metadata is empty. Aborting save to prevent inconsistent state.")
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}")

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)
        logger.info(f"FAISS metadata saved to {self.metadata_path}")

    def load_store(self):
        """Loads the FAISS index and metadata from files."""
        self.index = faiss.read_index(self.index_path)
        self.dimension = self.index.d
        logger.info(f"FAISS index loaded. Index has {self.index.ntotal} vectors of dim {self.index.d}.")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"FAISS metadata loaded. {len(self.metadata)} items.")

        if self.index.ntotal != len(self.metadata):
            logger.warning(f"Mismatch between FAISS index size ({self.index.ntotal}) and metadata size ({len(self.metadata)}).")

    def search(self, query_vector, k=5):
        """Performs a K-Nearest Neighbors search on the FAISS index."""
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot perform search.")
            return np.array([]), np.array([]), []

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype='float32')
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.index.d}).")

        distances, faiss_indices = self.index.search(query_vector, k)
        
        all_results_metadata = []
        for i in range(faiss_indices.shape[0]):
            query_results_metadata = []
            for j in range(faiss_indices.shape[1]):
                idx = faiss_indices[i][j]
                if idx != -1 and idx < len(self.metadata):
                    query_results_metadata.append(self.metadata[idx])
            all_results_metadata.append(query_results_metadata)
        
        return (distances[0], faiss_indices[0], all_results_metadata[0]) if query_vector.shape[0] == 1 else (distances, faiss_indices, all_results_metadata)

# --- Updated build script ---
if __name__ == '__main__':
    # Add project root to be able to import config
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    from src.config import ONTOLOGIES_CONFIG
    import pathlib

    logger.info("--- Building FAISS Stores for All Configured Ontologies ---")

    for name, config_data in ONTOLOGIES_CONFIG.items():
        index_path = config_data.get('faiss_index_path')
        metadata_path = config_data.get('faiss_metadata_path')
        embeddings_path = config_data.get('embeddings_path')

        logger.info(f"\n--- Processing Ontology: {name} ---")

        if not all([index_path, metadata_path, embeddings_path]):
            logger.warning(f"Configuration for '{name}' is missing one or more FAISS paths. Skipping.")
            continue

        pathlib.Path(index_path).unlink(missing_ok=True)
        pathlib.Path(metadata_path).unlink(missing_ok=True)
        logger.info(f"Removed old index files for '{name}' to ensure fresh build.")

        if not os.path.exists(embeddings_path):
            logger.error(f"ERROR: Embeddings file not found at {embeddings_path}. Cannot build FAISS index for '{name}'. Skipping.")
            continue

        store = FAISSVectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
            embeddings_file_path=embeddings_path
        )

        if store.index:
            logger.info(f"Successfully built and saved FAISS index for '{name}' with {store.index.ntotal} vectors.")
        else:
            logger.error(f"FAILED to build FAISS index for '{name}'. Check logs for errors.")
    
    logger.info("\n--- All FAISS Store Building Complete ---")