# src/retriever/hybrid_retriever.py
import json
import os
# import numpy as np # numpy is used by sentence_transformers and faiss internally
from whoosh.index import open_dir as open_whoosh_index
from whoosh.qparser import MultifieldParser, OrGroup
from sentence_transformers import SentenceTransformer

# --- Add project root to sys.path if running script directly ---
# This block is useful if you ever run this script directly (e.g., for debugging)
# and not as a module (python -m src.retriever.hybrid_retriever)
if __name__ == '__main__':
    import sys
    PROJECT_ROOT_FOR_DIRECT_RUN = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if PROJECT_ROOT_FOR_DIRECT_RUN not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_DIRECT_RUN)
# --- End sys.path modification ---


from src.vector_store.faiss_store import FAISSVectorStore
from src.config import (
    ONTOLOGY_DUMP_JSON,
    WHOOSH_INDEX_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    DEFAULT_K_LEXICAL,
    DEFAULT_K_VECTOR,
)
# from src.utils.logger import get_logger # Placeholder for future logging
# logger = get_logger(__name__) # Placeholder

class HybridRetriever:
    def __init__(self,
                 ontology_data_path=ONTOLOGY_DUMP_JSON,
                 whoosh_index_dir=WHOOSH_INDEX_DIR,
                 faiss_index_path=FAISS_INDEX_PATH,
                 faiss_metadata_path=FAISS_METADATA_PATH,
                 embedding_model_name=EMBEDDING_MODEL_NAME):
        """
        Initializes the HybridRetriever.
        """
        print(f"Initializing HybridRetriever...")
        # logger.info("Initializing HybridRetriever...")

        print(f"Loading ontology data from: {ontology_data_path}")
        if not os.path.exists(ontology_data_path):
            raise FileNotFoundError(f"Ontology data file not found: {ontology_data_path}")
        with open(ontology_data_path, 'r', encoding='utf-8') as f:
            self.ontology_data = json.load(f)
        print(f"Loaded {len(self.ontology_data)} ontology entries.")

        print(f"Loading Whoosh index from: {whoosh_index_dir}")
        if not os.path.exists(whoosh_index_dir) or not os.listdir(whoosh_index_dir):
            raise FileNotFoundError(f"Whoosh index directory not found or empty: {whoosh_index_dir}. Run ingestion scripts.")
        self.whoosh_ix = open_whoosh_index(whoosh_index_dir)
        self.whoosh_searcher = self.whoosh_ix.searcher()
        
        # Fields to search in Whoosh, must match the schema in build_lexical_index.py
        # 'relations_text' is indexed (stored=False) so it can be searched.
        # 'curie' is an ID field, typically not directly searched with MultifieldParser unless intended.
        self.whoosh_fields_to_search = ["label", "synonyms", "definition", "relations_text"]
        self.whoosh_parser = MultifieldParser(self.whoosh_fields_to_search, schema=self.whoosh_ix.schema, group=OrGroup)
        print("Whoosh index loaded.")

        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        print("Embedding model loaded.")

        print(f"Initializing FAISS vector store (index: {faiss_index_path}, metadata: {faiss_metadata_path})...")
        self.faiss_store = FAISSVectorStore(
            index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
            embeddings_file_path=None 
        )
        if not self.faiss_store.index or not self.faiss_store.metadata:
            raise FileNotFoundError(f"FAISS index file '{faiss_index_path}' or metadata file '{faiss_metadata_path}' not found or empty. Please build it first.")
        print("FAISS vector store initialized.")
        
        print("HybridRetriever initialized successfully.")

    def _lexical_search(self, query_string, limit=DEFAULT_K_LEXICAL):
        """
        Performs lexical search using Whoosh.
        Returns a list of dicts: {'id': str, 'label': str, 'score': float, 'source': 'lexical', 'details': dict}
        """
        results = []
        if not query_string:
            return results

        try:
            query = self.whoosh_parser.parse(query_string)
            search_results = self.whoosh_searcher.search(query, limit=limit)
            
            for hit in search_results:
                hit_fields = hit.fields()  # Get all stored fields as a dictionary
                term_curie = hit_fields.get('curie') # *** CHANGED: 'id' to 'curie' ***

                if term_curie is None:
                    print(f"Warning: Lexical search hit found without a 'curie'. Hit details: {hit}")
                    # logger.warning(f"Lexical search hit found without a 'curie'. Hit details: {hit}")
                    continue

                # 'relations_text' is not stored, so it won't be in hit_fields.
                # We retrieve label, synonyms, definition if they were stored.
                retrieved_label = hit_fields.get('label')
                retrieved_synonyms_str = hit_fields.get('synonyms') # This will be a space-separated string
                retrieved_definition = hit_fields.get('definition')

                results.append({
                    "id": term_curie, # Keep 'id' as the key in the result for consistency with vector search
                    "label": retrieved_label if retrieved_label is not None else self.ontology_data.get(term_curie, {}).get('label', 'N/A'),
                    "score": hit.score, 
                    "source": "lexical",
                    "details": {
                        # Convert synonyms string back to list if needed, or keep as string
                        "retrieved_synonyms": retrieved_synonyms_str.split() if retrieved_synonyms_str else [], 
                        "retrieved_definition": retrieved_definition
                    }
                })
        except Exception as e:
            print(f"Error during lexical search for '{query_string}': {e}")
            # logger.error(f"Error during lexical search for '{query_string}': {e}", exc_info=True)
            import traceback
            traceback.print_exc() # Print full traceback for debugging
        return results

    def _vector_search(self, query_string, k=DEFAULT_K_VECTOR):
        """
        Performs vector search using FAISS.
        Returns a list of dicts: {'id': str, 'label': str, 'score': float, 'source': 'vector', 'details': dict}
        """
        results = []
        if not query_string:
            return results

        try:
            query_vector = self.embedding_model.encode([query_string], convert_to_numpy=True)
            distances, _, metadata_items = self.faiss_store.search(query_vector, k=k)
            
            for i in range(len(metadata_items)):
                term_id = metadata_items[i]['id'] # FAISS metadata stores 'id'
                results.append({
                    "id": term_id,
                    "label": metadata_items[i]['label'],
                    "score": float(distances[i]), 
                    "source": "vector",
                    "details": {}
                })
        except Exception as e:
            print(f"Error during vector search for '{query_string}': {e}")
            # logger.error(f"Error during vector search for '{query_string}': {e}", exc_info=True)
            import traceback
            traceback.print_exc() # Print full traceback for debugging
        return results

    def search(self, query_string, lexical_limit=DEFAULT_K_LEXICAL, vector_k=DEFAULT_K_VECTOR):
        """
        Performs hybrid search.
        """
        lexical_results = self._lexical_search(query_string, limit=lexical_limit)
        vector_results = self._vector_search(query_string, k=vector_k)
        
        return {
            "query": query_string,
            "lexical_results": lexical_results,
            "vector_results": vector_results,
        }

    def get_term_details(self, term_id):
        """
        Retrieves full details for a given term ID (CURIE) from the loaded ontology data.
        """
        return self.ontology_data.get(term_id)

    def close(self):
        """
        Closes any open resources, like the Whoosh searcher.
        """
        if self.whoosh_searcher:
            self.whoosh_searcher.close()
        print("HybridRetriever resources closed.")

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This sys.path modification is now at the top of the file for when __name__ == '__main__'
    
    from src.config import PROJECT_ROOT # Import after sys.path is potentially modified
    print(f"Configured project root: {PROJECT_ROOT}")
    if not os.getcwd().startswith(PROJECT_ROOT) and os.getcwd() != PROJECT_ROOT:
         print(f"Warning: Current working directory ({os.getcwd()}) might not be the project root.")
         print("Consider running with 'python -m src.retriever.hybrid_retriever' from the project root directory.")

    print("Running HybridRetriever example...")
    retriever = None
    try:
        retriever = HybridRetriever()
        
        queries = ["GARLIC", "SALT", "GARBANZO", "TAHINI", "LEMON JUICE", "HONEY" ,"WATER", "OLIVE OIL", "ROSMARY", "HUMMUS"]
        
        for query in queries:
            print(f"\nSearching for: '{query}'")
            results = retriever.search(query, lexical_limit=3, vector_k=3)
            
            print("\n--- Lexical Results ---")
            if results["lexical_results"]:
                for res in results["lexical_results"]:
                    print(f"  ID: {res['id']}, Label: {res['label']}, Score (Whoosh): {res['score']:.4f}")
                    # print(f"    Details: {res['details']}") # Uncomment to see retrieved synonyms/def
            else:
                print("  No lexical results.")

            print("\n--- Vector Results ---")
            if results["vector_results"]:
                for res in results["vector_results"]:
                    print(f"  ID: {res['id']}, Label: {res['label']}, Score (L2 Distance): {res['score']:.4f}")
            else:
                print("  No vector results.")
            print("-" * 40)
            
    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found: {e}")
        print("Please ensure all data files (ontology_dump.json) and indices (Whoosh, FAISS) are correctly built and paths are set in src/config.py.")
        print("You might need to run the ingestion and embedding scripts first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if retriever:
            retriever.close()