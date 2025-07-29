# src/retriever/hybrid_retriever.py
import json
from pathlib import Path
from whoosh.index import open_dir as open_whoosh_index
from whoosh.qparser import MultifieldParser, OrGroup
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional, Dict, Any

# --- Add project root to sys.path if running script directly ---
if __name__ == '__main__':
    import sys
    # Use pathlib to determine the project root and add it to the path
    PROJECT_ROOT_FOR_DIRECT_RUN = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT_FOR_DIRECT_RUN) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_DIRECT_RUN))
# --- End sys.path modification ---

from src.vector_store.faiss_store import FAISSVectorStore
from src.config import (
    ONTOLOGIES_CONFIG,
    EMBEDDING_MODEL_NAME,
    DEFAULT_K_LEXICAL,
    DEFAULT_K_VECTOR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        """
        Initializes the HybridRetriever to work with multiple, separate ontologies
        defined in ONTOLOGIES_CONFIG.
        """
        logger.info("Initializing HybridRetriever for multiple ontologies...")
        
        self.ontology_data_stores: Dict[str, Dict[str, Any]] = {}
        self.whoosh_searchers: Dict[str, Any] = {}
        self.whoosh_parsers: Dict[str, MultifieldParser] = {}
        self.faiss_stores: Dict[str, FAISSVectorStore] = {}
        self.ontology_names = list(ONTOLOGIES_CONFIG.keys())
        self.prefix_to_name_map = {v['prefix']: k for k, v in ONTOLOGIES_CONFIG.items()}

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        
        for name, config_data in ONTOLOGIES_CONFIG.items():
            logger.info(f"--- Initializing resources for ontology: '{name}' ---")
            
            # Load ontology data dump using pathlib.Path
            dump_path = config_data['dump_json_path']
            logger.info(f"Loading ontology data from: {dump_path}")
            if not dump_path.exists():
                raise FileNotFoundError(f"Ontology dump for '{name}' not found: {dump_path}. Please run ingestion scripts.")
            with dump_path.open('r', encoding='utf-8') as f:
                self.ontology_data_stores[name] = json.load(f)
            logger.info(f"Loaded {len(self.ontology_data_stores[name])} entries for '{name}'.")

            # Load Whoosh index using pathlib.Path
            whoosh_dir = config_data['whoosh_index_dir']
            logger.info(f"Loading Whoosh index from: {whoosh_dir}")
            # Check if directory exists and is not empty
            if not whoosh_dir.is_dir() or not any(whoosh_dir.iterdir()):
                raise FileNotFoundError(f"Whoosh index for '{name}' not found or empty: {whoosh_dir}. Please run ingestion scripts.")
            
            # whoosh.index.open_dir expects a string path
            whoosh_ix = open_whoosh_index(str(whoosh_dir))
            self.whoosh_searchers[name] = whoosh_ix.searcher()
            whoosh_fields = ["label", "synonyms", "definition", "relations_text"]
            self.whoosh_parsers[name] = MultifieldParser(whoosh_fields, schema=whoosh_ix.schema, group=OrGroup)
            logger.info(f"Whoosh index for '{name}' loaded.")

            # Initialize FAISS store using pathlib.Path
            faiss_index_path = config_data['faiss_index_path']
            faiss_metadata_path = config_data['faiss_metadata_path']
            logger.info(f"Initializing FAISS store for '{name}' (index: {faiss_index_path}, metadata: {faiss_metadata_path})...")
            
            # Assuming FAISSVectorStore constructor accepts pathlib.Path objects
            faiss_store = FAISSVectorStore(
                index_path=faiss_index_path,
                metadata_path=faiss_metadata_path,
                embeddings_file_path=None # Not needed for loading
            )
            if not faiss_store.index or not faiss_store.metadata:
                 raise FileNotFoundError(f"FAISS index or metadata for '{name}' not found at {faiss_index_path}. Please build it first.")
            self.faiss_stores[name] = faiss_store
            logger.info(f"FAISS store for '{name}' initialized.")

        logger.info("HybridRetriever initialized successfully for all configured ontologies.")

    def _get_stores_to_query(self, store_dict: dict, target_ontologies: Optional[List[str]]):
        """Helper to select which stores (Whoosh/FAISS) to query."""
        if target_ontologies is None:
            # If no specific targets, use all available stores
            return store_dict.items()
        
        # Filter to only the targeted stores that actually exist
        stores_to_query = []
        for name in target_ontologies:
            if name in store_dict:
                stores_to_query.append((name, store_dict[name]))
            else:
                logger.warning(f"Requested ontology '{name}' not found in available stores. It will be skipped.")
        return stores_to_query

    def _lexical_search(self, query_string: str, limit: int = DEFAULT_K_LEXICAL, target_ontologies: Optional[List[str]] = None):
        """Performs lexical search on all or a subset of Whoosh indexes."""
        all_results = []
        if not query_string:
            return all_results

        searchers_to_query = self._get_stores_to_query(self.whoosh_searchers, target_ontologies)
        if not searchers_to_query:
            logger.warning("Lexical search: No valid target ontologies specified or found.")
            return []

        for name, searcher in searchers_to_query:
            try:
                parser = self.whoosh_parsers[name]
                query = parser.parse(query_string)
                search_results = searcher.search(query, limit=limit)
                
                for hit in search_results:
                    hit_fields = hit.fields()
                    term_curie = hit_fields.get('curie') 
                    if not term_curie: continue
                    
                    all_results.append({
                        "id": term_curie, "label": hit_fields.get('label', 'N/A'),
                        "score": hit.score, "source": "lexical", "source_ontology": name,
                    })
            except Exception as e:
                logger.error(f"Error during lexical search in '{name}' for '{query_string}': {e}", exc_info=True)
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results

    def _vector_search(self, query_string: str, k: int = DEFAULT_K_VECTOR, target_ontologies: Optional[List[str]] = None):
        """Performs vector search on all or a subset of FAISS indexes."""
        all_results = []
        if not query_string:
            return all_results

        stores_to_query = self._get_stores_to_query(self.faiss_stores, target_ontologies)
        if not stores_to_query:
            logger.warning("Vector search: No valid target ontologies specified or found.")
            return []

        try:
            query_vector = self.embedding_model.encode([query_string], convert_to_numpy=True)
            
            for name, store in stores_to_query:
                distances, _, metadata_items = store.search(query_vector, k=k)
                for i, item in enumerate(metadata_items):
                    all_results.append({
                        "id": item['id'], "label": item['label'], "score": float(distances[i]),
                        "source": "vector", "source_ontology": name,
                    })
        except Exception as e:
            logger.error(f"Error during vector search for '{query_string}': {e}", exc_info=True)

        all_results.sort(key=lambda x: x['score'])
        return all_results

    def search(self, query_string: str, lexical_limit: int = DEFAULT_K_LEXICAL, vector_k: int = DEFAULT_K_VECTOR, target_ontologies: Optional[List[str]] = None):
        """
        Performs hybrid search on all or a targeted subset of ontologies.

        Args:
            query_string (str): The search query.
            lexical_limit (int): Max number of results from lexical search.
            vector_k (int): Max number of results from vector search.
            target_ontologies (List[str], optional): A list of ontology names to search.
                                                     If None, searches all ontologies.
        """
        lexical_results = self._lexical_search(query_string, limit=lexical_limit, target_ontologies=target_ontologies)
        vector_results = self._vector_search(query_string, k=vector_k, target_ontologies=target_ontologies)
        
        return {
            "query": query_string,
            "lexical_results": lexical_results,
            "vector_results": vector_results,
        }

    def get_term_details(self, term_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full details for a given term ID by finding which ontology it belongs to.
        """
        matched_prefix = None
        # Find the longest matching prefix to handle overlapping namespaces (e.g., 'obo' and 'obo:FOODON_')
        for prefix in self.prefix_to_name_map.keys():
            if term_id.startswith(prefix):
                if matched_prefix is None or len(prefix) > len(matched_prefix):
                    matched_prefix = prefix
        
        if not matched_prefix:
            logger.warning(f"Could not determine ontology for term_id '{term_id}'.")
            return None
            
        ontology_name = self.prefix_to_name_map[matched_prefix]
        term_data = self.ontology_data_stores.get(ontology_name, {}).get(term_id)
        
        if term_data:
            # Return a copy to prevent modification of the original loaded data
            term_data = dict(term_data)
            term_data['id'] = term_id
        return term_data

    def close(self):
        """Closes any open resources, such as Whoosh searchers."""
        for name, searcher in self.whoosh_searchers.items():
            if searcher:
                searcher.close()
                logger.info(f"Whoosh searcher for '{name}' closed.")

# --- The example usage block remains unchanged but now works with the refactored code ---
if __name__ == '__main__':
    logger.info("Running HybridRetriever example...")
    retriever = None
    try:
        retriever = HybridRetriever()
        
        print("\n\n" + "="*80)
        print("✅ 1. Searching for 'cheese' across ALL ontologies (default behavior)")
        print("="*80)
        results = retriever.search("cheese", lexical_limit=2, vector_k=2)
        print(json.dumps(results, indent=2))

        print("\n\n" + "="*80)
        print("✅ 2. Searching for 'cheese' ONLY in the 'foodon' ontology")
        print("="*80)
        results_foodon = retriever.search("cheese", lexical_limit=2, vector_k=2, target_ontologies=["foodon"])
        print(json.dumps(results_foodon, indent=2))
        
        print("\n\n" + "="*80)
        print("✅ 3. Searching for 'chemical entity' ONLY in the 'chebi' ontology")
        print("="*80)
        results_chebi = retriever.search("chemical entity", lexical_limit=2, vector_k=2, target_ontologies=["chebi"])
        print(json.dumps(results_chebi, indent=2))

        print("\n\n" + "="*80)
        print("❌ 4. Searching with an invalid ontology name (should be skipped gracefully)")
        print("="*80)
        results_invalid = retriever.search("cheese", lexical_limit=2, vector_k=2, target_ontologies=["non_existent_ontology"])
        print(json.dumps(results_invalid, indent=2))

    except FileNotFoundError as e:
        logger.error(f"\nERROR: A required file was not found: {e}. Please ensure you have run the data ingestion and indexing scripts first.")
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during example run: {e}", exc_info=True)
    finally:
        if retriever:
            retriever.close()