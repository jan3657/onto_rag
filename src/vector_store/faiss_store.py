# src/vector_store/faiss_store.py
import faiss
import json
import numpy as np
import os
from src.config import EMBEDDINGS_FILE # Default embeddings file to build from

# from src.utils.logger import get_logger # Placeholder for future logging
# logger = get_logger(__name__)

class FAISSVectorStore:
    def __init__(self, index_path, metadata_path, embeddings_file_path=None, dimension=None):
        """
        Initializes the FAISSVectorStore.
        Tries to load an existing index and metadata. If not found, and an
        embeddings_file_path is provided, it will attempt to build them.

        Args:
            index_path (str): Path to save/load the FAISS index file (.bin).
            metadata_path (str): Path to save/load the metadata JSON file (.json).
            embeddings_file_path (str, optional): Path to the 'embeddings.json' file.
                                                  Used only if the index/metadata needs to be built.
                                                  Defaults to config.EMBEDDINGS_FILE.
            dimension (int, optional): The dimension of the vectors.
                                       Required if building the index and it cannot be inferred.
                                       Usually inferred from the first embedding.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embeddings_file_path = embeddings_file_path if embeddings_file_path else EMBEDDINGS_FILE
        self.dimension = dimension

        self.index = None
        self.metadata = []  # List of dicts, e.g., [{'id': 'FOODON_123', 'label': 'Apple'}, ...]

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print(f"Loading existing FAISS index from {self.index_path} and metadata from {self.metadata_path}")
            # logger.info(f"Loading existing FAISS index from {self.index_path} and metadata from {self.metadata_path}")
            try:
                self.load_store()
            except Exception as e:
                print(f"Error loading existing FAISS store: {e}. Will attempt to build if embeddings file provided.")
                # logger.error(f"Error loading existing FAISS store: {e}. Will attempt to build if embeddings file provided.", exc_info=True)
                self._try_build_store()
        elif self.embeddings_file_path and os.path.exists(self.embeddings_file_path):
            print(f"FAISS index/metadata not found. Attempting to build from {self.embeddings_file_path}")
            # logger.info(f"FAISS index/metadata not found. Attempting to build from {self.embeddings_file_path}")
            self._try_build_store()
        else:
            message = "FAISS index/metadata not found. "
            if self.embeddings_file_path:
                message += f"Embeddings file {self.embeddings_file_path} also not found or not specified for building."
            else:
                message += "No embeddings file path provided to build a new store."
            print(f"Warning: {message}")
            # logger.warning(message)

    def _try_build_store(self):
        """Helper method to attempt building the store."""
        embeddings_data = self._load_embeddings_data()
        if embeddings_data:
            self.build_index_from_embeddings(embeddings_data)
            if self.index and self.metadata: # Check if build was successful
                 self.save_store()
            else:
                print("Warning: FAISS index or metadata not built successfully from embeddings data.")
                # logger.warning("FAISS index or metadata not built successfully from embeddings data.")
        else:
            print(f"Warning: No embeddings data loaded from {self.embeddings_file_path}, FAISS index not built.")
            # logger.warning(f"No embeddings data loaded from {self.embeddings_file_path}, FAISS index not built.")


    def _load_embeddings_data(self):
        """
        Loads embeddings data from the specified JSON file.
        Expected format: list of {'id': str, 'label': str, 'embedding': list[float]}
        """
        if not self.embeddings_file_path or not os.path.exists(self.embeddings_file_path):
            print(f"Error: Embeddings file not found: {self.embeddings_file_path}")
            # logger.error(f"Embeddings file not found: {self.embeddings_file_path}")
            return None
        try:
            with open(self.embeddings_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Sanity check the data structure
            if not isinstance(data, list) or not data:
                print(f"Error: Embeddings file {self.embeddings_file_path} is empty or not a list.")
                # logger.error(f"Embeddings file {self.embeddings_file_path} is empty or not a list.")
                return None
            if not all('id' in item and 'label' in item and 'embedding' in item for item in data):
                print(f"Error: Embeddings data in {self.embeddings_file_path} has incorrect format.")
                # logger.error(f"Embeddings data in {self.embeddings_file_path} has incorrect format.")
                return None

            print(f"Loaded {len(data)} embeddings from {self.embeddings_file_path}")
            # logger.info(f"Loaded {len(data)} embeddings from {self.embeddings_file_path}")
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from embeddings file {self.embeddings_file_path}: {e}")
            # logger.error(f"Error decoding JSON from embeddings file {self.embeddings_file_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading embeddings file {self.embeddings_file_path}: {e}")
            # logger.error(f"An unexpected error occurred while loading embeddings file {self.embeddings_file_path}: {e}", exc_info=True)
            return None

    def build_index_from_embeddings(self, embeddings_data):
        """
        Builds the FAISS index and prepares metadata from loaded embeddings data.

        Args:
            embeddings_data (list): List of dictionaries, where each dict has
                                    'id', 'label', and 'embedding' keys.
        """
        if not embeddings_data:
            print("Warning: No embeddings data provided to build_index_from_embeddings.")
            # logger.warning("No embeddings data provided to build_index_from_embeddings.")
            return

        vectors = []
        current_metadata = [] # Use a temporary list to build metadata
        for item in embeddings_data:
            vectors.append(item['embedding'])
            current_metadata.append({'id': item['id'], 'label': item['label']})

        if not vectors:
            print("Warning: No vectors extracted from embeddings_data.")
            # logger.warning("No vectors extracted from embeddings_data.")
            return

        vectors_np = np.array(vectors).astype('float32')

        if self.dimension is None:
            self.dimension = vectors_np.shape[1]
        elif self.dimension != vectors_np.shape[1]:
            err_msg = f"Provided dimension {self.dimension} does not match embeddings dimension {vectors_np.shape[1]}"
            print(f"Error: {err_msg}")
            # logger.error(err_msg)
            raise ValueError(err_msg)

        # Using IndexFlatL2, a simple L2 distance index.
        # For larger datasets, more advanced indexes like IndexIVFFlat might be better.
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_np)
            self.metadata = current_metadata # Assign once build is successful
            print(f"FAISS index built successfully with {self.index.ntotal} vectors of dimension {self.dimension}.")
            # logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors of dimension {self.dimension}.")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            # logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.index = None # Ensure index is None if build fails
            self.metadata = []


    def save_store(self):
        """Saves the FAISS index and metadata to their respective files."""
        if self.index is None:
            print("Warning: No FAISS index to save.")
            # logger.warning("No FAISS index to save.")
            return # Do not save metadata if index is not there or failed to build

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")
        # logger.info(f"FAISS index saved to {self.index_path}")

        if not self.metadata:
            print("Warning: No FAISS metadata to save (metadata list is empty).")
            # logger.warning("No FAISS metadata to save (metadata list is empty).")
            # If index exists but metadata is empty, this is an inconsistent state.
            # Depending on strictness, one might choose to not save the index either,
            # or clear the index file if it exists. For now, we save index if it exists.
            return

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)
        print(f"FAISS metadata saved to {self.metadata_path}")
        # logger.info(f"FAISS metadata saved to {self.metadata_path}")

    def load_store(self):
        """Loads the FAISS index and metadata from files."""
        if not os.path.exists(self.index_path):
            # logger.error(f"FAISS index file not found: {self.index_path}")
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        if self.dimension is None: # Infer dimension if not set
            self.dimension = self.index.d
        elif self.dimension != self.index.d: # Check consistency
            # logger.warning(f"Stored index dimension {self.index.d} differs from configured {self.dimension}. Using stored.")
            print(f"Warning: Stored index dimension {self.index.d} differs from configured {self.dimension}. Using stored.")
            self.dimension = self.index.d

        print(f"FAISS index loaded from {self.index_path}. Index has {self.index.ntotal} vectors of dim {self.index.d}.")
        # logger.info(f"FAISS index loaded from {self.index_path}. Index has {self.index.ntotal} vectors of dim {self.index.d}.")

        if not os.path.exists(self.metadata_path):
            # logger.error(f"FAISS metadata file not found: {self.metadata_path}")
            raise FileNotFoundError(f"FAISS metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"FAISS metadata loaded from {self.metadata_path}. {len(self.metadata)} items.")
        # logger.info(f"FAISS metadata loaded from {self.metadata_path}. {len(self.metadata)} items.")

        if self.index and self.metadata and self.index.ntotal != len(self.metadata):
            mismatch_msg = (f"Mismatch between FAISS index size ({self.index.ntotal}) "
                            f"and metadata size ({len(self.metadata)}). This may lead to errors.")
            print(f"Warning: {mismatch_msg}")
            # logger.warning(mismatch_msg)
            # Potentially raise an error or try to reconcile, for now, just warn.

    def search(self, query_vector, k=5):
        """
        Performs a K-Nearest Neighbors search on the FAISS index.

        Args:
            query_vector (np.ndarray): A 2D numpy array of shape (num_queries, dimension)
                                       or a 1D numpy array (single query).
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: (distances, indices, metadata_items)
                   distances (np.ndarray): Distances to the k nearest neighbors.
                   indices (np.ndarray): FAISS internal indices of the k nearest neighbors.
                   metadata_items (list of lists or list of dicts): Corresponding metadata for the results.
                                     If single query, returns list of dicts. If multiple, list of lists of dicts.
        """
        if self.index is None:
            print("Error: FAISS index not initialized. Cannot perform search.")
            # logger.error("FAISS index not initialized. Cannot perform search.")
            return np.array([]), np.array([]), []

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)

        if query_vector.ndim == 1: # Single query vector
            query_vector = np.expand_dims(query_vector, axis=0)
        
        if query_vector.shape[1] != self.index.d:
            err_msg = f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.index.d})."
            print(f"Error: {err_msg}")
            # logger.error(err_msg)
            raise ValueError(err_msg)

        distances, faiss_indices = self.index.search(query_vector.astype('float32'), k)
        
        # faiss_indices will be shape (num_queries, k)
        # distances will be shape (num_queries, k)
        
        all_results_metadata = []
        for i in range(faiss_indices.shape[0]): # Iterate over queries
            query_results_metadata = []
            for j in range(faiss_indices.shape[1]): # Iterate over k results for that query
                idx = faiss_indices[i][j]
                if idx != -1 and idx < len(self.metadata): # faiss_index can be -1 if k > ntotal
                    query_results_metadata.append(self.metadata[idx])
                # else: could append a placeholder, or log. For now, items are just shorter if fewer than k found.
            all_results_metadata.append(query_results_metadata)
        
        # For a single query_vector (most common use case here), return the inner list directly.
        if query_vector.shape[0] == 1:
            return distances[0], faiss_indices[0], all_results_metadata[0]
        else: # If multiple query vectors were passed
            return distances, faiss_indices, all_results_metadata

# Example usage / test function
if __name__ == '__main__':
    from src.config import DATA_DIR, FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDINGS_FILE

    print("Running FAISSVectorStore example...")

    # --- Test Case 1: Building the index if it doesn't exist ---
    # For this test, we'll create a dummy embeddings.json if it doesn't exist
    # Ensure EMBEDDINGS_FILE path is correct in config
    
    # Clean up old test files if they exist, to ensure a clean build test
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        print(f"Removed existing test index: {FAISS_INDEX_PATH}")
    if os.path.exists(FAISS_METADATA_PATH):
        os.remove(FAISS_METADATA_PATH)
        print(f"Removed existing test metadata: {FAISS_METADATA_PATH}")

    # Create a dummy embeddings.json if it does not exist or is invalid
    dummy_embeddings_data = [
        {"id": "FOODON:001", "label": "Test Apple", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"id": "FOODON:002", "label": "Test Banana", "embedding": [0.5, 0.6, 0.7, 0.8]},
        {"id": "FOODON:003", "label": "Test Orange", "embedding": [0.2, 0.3, 0.4, 0.5]},
        {"id": "FOODON:004", "label": "Test Grape", "embedding": [0.6, 0.7, 0.8, 0.9]},
        {"id": "FOODON:005", "label": "Another Apple", "embedding": [0.11, 0.21, 0.31, 0.41]}, # Similar to Test Apple
    ]
    vector_dim = len(dummy_embeddings_data[0]["embedding"])

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Creating dummy embeddings file: {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(dummy_embeddings_data, f, indent=4)
    else:
        print(f"Using existing embeddings file: {EMBEDDINGS_FILE}")
        # Optionally, load and check its dimension here or rely on FAISSVectorStore to do it.

    # Initialize FAISSVectorStore - this should build and save if files are missing
    print("\n--- Initializing FAISS store (should build if not present) ---")
    try:
        faiss_store = FAISSVectorStore(
            index_path=FAISS_INDEX_PATH,
            metadata_path=FAISS_METADATA_PATH,
            embeddings_file_path=EMBEDDINGS_FILE, # Explicitly pass for build
            dimension=vector_dim # Provide dimension if known, or let it infer
        )

        if faiss_store.index and faiss_store.metadata:
            print("FAISS store initialized/built successfully.")
            print(f"Index has {faiss_store.index.ntotal} items.")

            # --- Test Case 2: Searching the index ---
            print("\n--- Testing Search ---")
            # Query for something similar to "Test Apple"
            query_vec = np.array([0.1, 0.2, 0.3, 0.4]).astype('float32')
            distances, indices, metadata_items = faiss_store.search(query_vec, k=3)

            print(f"Query vector: {query_vec}")
            print(f"Distances: {distances}")
            print(f"FAISS Indices: {indices}") # These are row numbers in the original embeddings list
            print("Metadata of results:")
            for item in metadata_items:
                print(f"  ID: {item['id']}, Label: {item['label']}")
            
            assert len(metadata_items) > 0, "Search returned no results"
            assert metadata_items[0]['id'] == "FOODON:001", "Top search result mismatch"

            # --- Test Case 3: Loading the existing index ---
            print("\n--- Testing Loading Existing Store ---")
            faiss_store_loaded = FAISSVectorStore(
                index_path=FAISS_INDEX_PATH,
                metadata_path=FAISS_METADATA_PATH
                # Not providing embeddings_file_path, so it must load
            )
            if faiss_store_loaded.index and faiss_store_loaded.metadata:
                print("FAISS store loaded successfully from disk.")
                print(f"Loaded index has {faiss_store_loaded.index.ntotal} items.")
                
                # Re-run search to confirm
                distances_loaded, _, metadata_items_loaded = faiss_store_loaded.search(query_vec, k=1)
                print("Search result from loaded store:")
                for item in metadata_items_loaded:
                    print(f"  ID: {item['id']}, Label: {item['label']}")
                assert metadata_items_loaded[0]['id'] == "FOODON:001", "Loaded store search result mismatch"
            else:
                print("Failed to load FAISS store from disk.")
        else:
            print("FAISS store could not be initialized or built.")

    except Exception as e:
        print(f"An error occurred during FAISSVectorStore example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files if you want to keep the data dir clean after tests
        # For this example, we'll leave them so you can inspect
        # if os.path.exists(EMBEDDINGS_FILE) and EMBEDDINGS_FILE.endswith("dummy_embeddings.json"):
        #     os.remove(EMBEDDINGS_FILE)
        # if os.path.exists(FAISS_INDEX_PATH) and FAISS_INDEX_PATH.endswith("dummy_faiss.bin"):
        #     os.remove(FAISS_INDEX_PATH)
        # if os.path.exists(FAISS_METADATA_PATH) and FAISS_METADATA_PATH.endswith("dummy_faiss_meta.json"):
        #     os.remove(FAISS_METADATA_PATH)
        pass