# infrastructure/retrieval/hybrid_retriever.py
import asyncio
import json
from whoosh.index import open_dir as open_whoosh_index
from whoosh.qparser import MultifieldParser, OrGroup
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional, Dict, Any

from src.components.faiss_store import FAISSVectorStore
from src import config  # Import module to access config dynamically

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self):
        """
        Initializes the HybridRetriever to work with multiple, separate ontologies
        defined in config.ONTOLOGIES_CONFIG.
        
        Note: Uses config.ONTOLOGIES_CONFIG dynamically so that evaluation scripts
        can override it before creating the pipeline.
        """
        logger.info("Initializing HybridRetriever for multiple ontologies...")
        
        self.ontology_data_stores: Dict[str, Dict[str, Any]] = {}
        # --- MODIFICATION 1: Store the main index objects, not the searchers ---
        self.whoosh_indexes: Dict[str, Any] = {}
        self.whoosh_parsers: Dict[str, MultifieldParser] = {}
        self.faiss_stores: Dict[str, FAISSVectorStore] = {}
        self.ontology_names = list(config.ONTOLOGIES_CONFIG.keys())
        self.prefix_to_name_map = {v['prefix']: k for k, v in config.ONTOLOGIES_CONFIG.items()}

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, trust_remote_code=True)
        
        for name, config_data in config.ONTOLOGIES_CONFIG.items():
            logger.info(f"--- Initializing resources for ontology: '{name}' ---")

            
            dump_path = config_data['dump_json_path']
            logger.info(f"Loading ontology data from: {dump_path}")
            if not dump_path.exists():
                raise FileNotFoundError(f"Ontology dump for '{name}' not found: {dump_path}.")
            with dump_path.open('r', encoding='utf-8') as f:
                self.ontology_data_stores[name] = json.load(f)
            logger.info(f"Loaded {len(self.ontology_data_stores[name])} entries for '{name}'.")

            whoosh_dir = config_data['whoosh_index_dir']
            logger.info(f"Loading Whoosh index from: {whoosh_dir}")
            if not whoosh_dir.is_dir() or not any(whoosh_dir.iterdir()):
                raise FileNotFoundError(f"Whoosh index for '{name}' not found or empty: {whoosh_dir}.")
            
            whoosh_ix = open_whoosh_index(str(whoosh_dir))
            # --- MODIFICATION 1 (cont.): Store the index itself ---
            self.whoosh_indexes[name] = whoosh_ix 
            whoosh_fields = ["label", "synonyms", "definition", "relations_text"]
            self.whoosh_parsers[name] = MultifieldParser(whoosh_fields, schema=whoosh_ix.schema, group=OrGroup)
            logger.info(f"Whoosh index for '{name}' loaded.")

            faiss_index_path = config_data['faiss_index_path']
            faiss_metadata_path = config_data['faiss_metadata_path']
            logger.info(f"Initializing FAISS store for '{name}'...")
            
            faiss_store = FAISSVectorStore(
                index_path=faiss_index_path,
                metadata_path=faiss_metadata_path,
                embeddings_file_path=None
            )
            if not faiss_store.index or not faiss_store.metadata:
                 raise FileNotFoundError(f"FAISS index or metadata for '{name}' not found at {faiss_index_path}.")
            self.faiss_stores[name] = faiss_store
            logger.info(f"FAISS store for '{name}' initialized.")

        logger.info("HybridRetriever initialized successfully for all configured ontologies.")

    def _get_stores_to_query(self, store_dict: dict, target_ontologies: Optional[List[str]]):
        if target_ontologies is None:
            return store_dict.items()
        
        stores_to_query = []
        for name in target_ontologies:
            if name in store_dict:
                stores_to_query.append((name, store_dict[name]))
            else:
                logger.warning(f"Requested ontology '{name}' not found. It will be skipped.")
        return stores_to_query

    # --- MODIFICATION 2: Complete overhaul of lexical_search for robustness ---
    def _lexical_search(self, query_string: str, limit: int = None, target_ontologies: Optional[List[str]] = None):
        limit = limit if limit is not None else config.DEFAULT_K_LEXICAL
        """Performs lexical search with detailed logging to pinpoint hangs."""
        logger.info(f"Starting lexical search for query: '{query_string}'")
        all_results = []
        if not query_string:
            return all_results

        indexes_to_query = self._get_stores_to_query(self.whoosh_indexes, target_ontologies)
        if not indexes_to_query:
            logger.warning("Lexical search: No valid target ontologies specified or found.")
            return []

        for name, ix in indexes_to_query:
            try:
                # --- LOGGING STEP 1: PARSING ---
                parser = self.whoosh_parsers[name]
                logger.info(f"Lexical search: About to parse query '{query_string}' for index '{name}'.")
                query = parser.parse(query_string)
                logger.info(f"Lexical search: Successfully parsed query for index '{name}'. Parsed query: {query}")

                with ix.searcher() as searcher:
                    # --- LOGGING STEP 2: SEARCHING ---
                    logger.info(f"Lexical search: About to execute search on index '{name}'.")
                    search_results = searcher.search(query, limit=limit)
                    logger.info(f"Lexical search: Search on index '{name}' completed. Found {len(search_results)} hits.")
                
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
        logger.info(f"Finished lexical search for query: '{query_string}'")
        return all_results
    
    def _vector_search(self, query_string: str, k: int = None, target_ontologies: Optional[List[str]] = None):
        k = k if k is not None else config.DEFAULT_K_VECTOR
        """Performs vector search on all or a subset of FAISS indexes."""
        logger.info(f"Starting vector search for query: '{query_string}'")
        all_results = []
        if not query_string:
            return all_results

        stores_to_query = self._get_stores_to_query(self.faiss_stores, target_ontologies)
        if not stores_to_query:
            logger.warning("Vector search: No valid target ontologies specified or found.")
            return []

        try:
            logger.info(f"Vector search: Encoding query '{query_string}'.")
            query_vector = self.embedding_model.encode([query_string], convert_to_numpy=True)
            logger.info(f"Vector search: Query encoding complete.")
            
            for name, store in stores_to_query:
                logger.info(f"Vector search: Querying FAISS index '{name}'")
                distances, _, metadata_items = store.search(query_vector, k=k)
                logger.info(f"Vector search: FAISS index '{name}' returned {len(metadata_items)} results.")
                for i, item in enumerate(metadata_items):
                    all_results.append({
                        "id": item['id'], "label": item['label'], "score": float(distances[i]),
                        "source": "vector", "source_ontology": name,
                    })
        except Exception as e:
            logger.error(f"Error during vector search for '{query_string}': {e}", exc_info=True)

        all_results.sort(key=lambda x: x['score'])
        if all_results:
            top_ids = [r['id'] for r in all_results[:10]]
            top_scores = [f"{r['score']:.4f}" for r in all_results[:10]]
            logger.debug(f"[RETRIEVER_VECTOR_RESULTS] query='{query_string}' | top_ids={top_ids} | top_scores={top_scores}")
        logger.info(f"Finished vector search for query: '{query_string}'")
        return all_results

    def search(self, query_string: str, lexical_limit: int = None, vector_k: int = None, target_ontologies: Optional[List[str]] = None, trace_id: str = None):
        lexical_limit = lexical_limit if lexical_limit is not None else config.DEFAULT_K_LEXICAL
        vector_k = vector_k if vector_k is not None else config.DEFAULT_K_VECTOR
        """
        Performs hybrid search with deterministic candidate deduplication.
        
        Deduplication: Merges by (ontology, id) key, keeping:
        - For lexical: highest score (BM25, higher is better)
        - For vector: lowest score (L2 distance, lower is better)
        - If same ID appears in both: keeps lexical (more precise match)
        """
        logger.info(f"Retriever.search initiated for query: '{query_string}'")

        lexical_results = self._lexical_search(query_string, limit=lexical_limit, target_ontologies=target_ontologies)
        vector_results = self._vector_search(query_string, k=vector_k, target_ontologies=target_ontologies)
        
        # --- Deterministic deduplication by (ontology, id) ---
        # Build lookup: key -> candidate, preferring lexical over vector for same ID
        seen: Dict[tuple, Dict[str, Any]] = {}
        
        # Process lexical first (higher priority)
        for cand in lexical_results:
            key = (cand.get("source_ontology", ""), cand.get("id", ""))
            if key not in seen:
                seen[key] = cand
            else:
                # Keep higher BM25 score
                if cand.get("score", 0) > seen[key].get("score", 0):
                    seen[key] = cand
        
        lexical_ids = set(seen.keys())
        
        # Process vector results
        for cand in vector_results:
            key = (cand.get("source_ontology", ""), cand.get("id", ""))
            if key not in seen:
                seen[key] = cand
            elif seen[key].get("source") == "vector":
                # Keep lower L2 distance
                if cand.get("score", float('inf')) < seen[key].get("score", float('inf')):
                    seen[key] = cand
            # If key exists from lexical, don't replace (lexical is more precise)
        
        vector_ids = {(c.get("source_ontology", ""), c.get("id", "")) for c in vector_results}
        overlap_count = len(lexical_ids & vector_ids)
        
        # Build final list, sorted by original retrieval order preference
        merged_candidates = list(seen.values())
        # Sort: lexical first (by descending score), then vector (by ascending score)
        merged_candidates.sort(key=lambda x: (
            0 if x.get("source") == "lexical" else 1,
            -x.get("score", 0) if x.get("source") == "lexical" else x.get("score", 0)
        ))
        
        # Log intermediate counts
        merge_stats = {
            "lexical_k": len(lexical_results),
            "vector_k": len(vector_results),
            "overlap": overlap_count,
            "unique_after_dedupe": len(merged_candidates),
            "final_k": len(merged_candidates),
        }
        logger.debug(f"[RETRIEVER_MERGE_STATS] query='{query_string}' | {json.dumps(merge_stats)}")
        
        # Log full candidate tuples
        candidate_tuples = [
            {"ontology": c.get("source_ontology"), "id": c.get("id"), "label": c.get("label"), 
             "score": round(c.get("score", 0), 4), "source": c.get("source")}
            for c in merged_candidates[:20]
        ]
        logger.debug(f"[RETRIEVER_CANDIDATES_FINAL] query='{query_string}' | candidates={json.dumps(candidate_tuples, ensure_ascii=False)}")
        
        logger.info(f"Retriever.search finished for query: '{query_string}' | {merge_stats}")
        return {
            "query": query_string,
            "lexical_results": lexical_results,
            "vector_results": vector_results,
            "merged_candidates": merged_candidates,
            "merge_stats": merge_stats,
        }
    
    async def search_async(self, query_string: str, lexical_limit: int = None, vector_k: int = None, target_ontologies: Optional[List[str]] = None, trace_id: str = None):
        """
        Async wrapper for search that runs blocking operations in a thread pool.
        This prevents blocking the event loop during CPU-bound embedding and index operations.
        """
        import asyncio
        return await asyncio.to_thread(
            self.search, query_string, lexical_limit, vector_k, target_ontologies, trace_id
        )

    def get_term_details(self, term_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full details for a given term ID.
        """
        matched_prefix = None
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
            term_data = dict(term_data)
            term_data['id'] = term_id
        return term_data

    # The close() method is no longer necessary as searchers are managed by 'with' statements.
    def close(self):
        """This method can be left empty or removed."""
        logger.info("Retriever resources are now managed automatically; close() is no longer required.")
        pass

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