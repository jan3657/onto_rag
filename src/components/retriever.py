# infrastructure/retrieval/hybrid_retriever.py
import asyncio
import json
from whoosh.index import open_dir as open_whoosh_index
from whoosh.qparser import MultifieldParser, OrGroup
# from sentence_transformers import SentenceTransformer (lazy loaded in __init__)
import logging
from typing import List, Optional, Dict, Any

from src.components.faiss_store import FAISSVectorStore
from src import config  # Import module to access config dynamically
from src.utils.model_loading import load_sentence_transformer_model
from src.utils.text_normalization import normalize_biomedical_text

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self):
        """
        Initializes the HybridRetriever to work with multiple, separate ontologies
        defined in config.ONTOLOGIES_CONFIG.

        Now supports 3 retrieval sources:
        - Lexical (Whoosh BM25)
        - Vector MiniLM (general semantic)
        - Vector SapBERT (biomedical-specific semantic)

        Note: Uses config.ONTOLOGIES_CONFIG dynamically so that evaluation scripts
        can override it before creating the pipeline.
        """
        logger.info("Initializing HybridRetriever for multiple ontologies...")

        self.ontology_data_stores: Dict[str, Dict[str, Any]] = {}
        self.whoosh_indexes: Dict[str, Any] = {}
        self.whoosh_parsers: Dict[str, MultifieldParser] = {}
        # Separate FAISS stores for each embedding model
        self.faiss_stores_minilm: Dict[str, FAISSVectorStore] = {}
        self.faiss_stores_sapbert: Dict[str, FAISSVectorStore] = {}
        self.ontology_names = list(config.ONTOLOGIES_CONFIG.keys())
        self.prefix_to_name_map = {v['prefix']: k for k, v in config.ONTOLOGIES_CONFIG.items()}

        # Load MiniLM embedding model
        # Lazy load to avoid startup cost
        logger.info(f"Loading MiniLM embedding model: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_model_minilm = load_sentence_transformer_model(
            config.EMBEDDING_MODEL_NAME,
        )

        # Load SapBERT embedding model
        logger.info(f"Loading SapBERT embedding model: {config.SAPBERT_MODEL_NAME}")
        self.embedding_model_sapbert = load_sentence_transformer_model(
            config.SAPBERT_MODEL_NAME,
        )

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
            whoosh_fields = [
                field_name
                for field_name in ["label", "synonyms", "definition", "relations_text"]
                if field_name in whoosh_ix.schema.names()
            ]
            if not whoosh_fields:
                raise ValueError(
                    f"Whoosh index for '{name}' has no searchable text fields in schema."
                )
            self.whoosh_parsers[name] = MultifieldParser(whoosh_fields, schema=whoosh_ix.schema, group=OrGroup)
            logger.info(f"Whoosh index for '{name}' loaded.")

            # Load MiniLM FAISS index (with backward compatibility fallback)
            faiss_index_minilm_path = config_data.get('faiss_index_minilm_path') or config_data.get('faiss_index_path')
            faiss_metadata_minilm_path = config_data.get('faiss_metadata_minilm_path') or config_data.get('faiss_metadata_path')
            logger.info(f"Initializing MiniLM FAISS store for '{name}'...")

            faiss_store_minilm = FAISSVectorStore(
                index_path=faiss_index_minilm_path,
                metadata_path=faiss_metadata_minilm_path,
                embeddings_file_path=None
            )
            if not faiss_store_minilm.index or not faiss_store_minilm.metadata:
                 raise FileNotFoundError(f"MiniLM FAISS index or metadata for '{name}' not found at {faiss_index_minilm_path}.")
            self.faiss_stores_minilm[name] = faiss_store_minilm
            logger.info(f"MiniLM FAISS store for '{name}' initialized.")

            # Load SapBERT FAISS index (optional for now - graceful degradation)
            faiss_index_sapbert_path = config_data.get('faiss_index_sapbert_path')
            faiss_metadata_sapbert_path = config_data.get('faiss_metadata_sapbert_path')

            if faiss_index_sapbert_path and faiss_index_sapbert_path.exists():
                logger.info(f"Initializing SapBERT FAISS store for '{name}'...")
                faiss_store_sapbert = FAISSVectorStore(
                    index_path=faiss_index_sapbert_path,
                    metadata_path=faiss_metadata_sapbert_path,
                    embeddings_file_path=None
                )
                if not faiss_store_sapbert.index or not faiss_store_sapbert.metadata:
                    logger.warning(f"SapBERT FAISS index or metadata for '{name}' not found at {faiss_index_sapbert_path}. SapBERT retrieval will be disabled for this ontology.")
                    self.faiss_stores_sapbert[name] = None
                else:
                    self.faiss_stores_sapbert[name] = faiss_store_sapbert
                    logger.info(f"SapBERT FAISS store for '{name}' initialized.")
            else:
                logger.warning(f"SapBERT FAISS paths not configured or files don't exist for '{name}'. SapBERT retrieval will be disabled for this ontology.")
                self.faiss_stores_sapbert[name] = None

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
                        label = hit_fields.get('label')
                        if not label:
                            label = (
                                self.ontology_data_stores
                                .get(name, {})
                                .get(term_curie, {})
                                .get("label", "N/A")
                            )

                        all_results.append({
                            "id": term_curie, "label": label,
                            "score": hit.score, "source": "lexical", "source_ontology": name,
                        })
            except Exception as e:
                logger.error(f"Error during lexical search in '{name}' for '{query_string}': {e}", exc_info=True)

        all_results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Finished lexical search for query: '{query_string}'")
        return all_results

    def _vector_search_minilm(self, query_string: str, k: int = None, target_ontologies: Optional[List[str]] = None):
        k = k if k is not None else config.DEFAULT_K_MINILM
        """Performs vector search using MiniLM embeddings."""
        logger.info(f"Starting MiniLM vector search for query: '{query_string}'")
        all_results = []
        if not query_string:
            return all_results

        stores_to_query = self._get_stores_to_query(self.faiss_stores_minilm, target_ontologies)
        if not stores_to_query:
            logger.warning("MiniLM vector search: No valid target ontologies specified or found.")
            return []

        try:
            logger.info(f"MiniLM vector search: Encoding query '{query_string}'.")
            query_vector = self.embedding_model_minilm.encode([query_string], convert_to_numpy=True)
            logger.info(f"MiniLM vector search: Query encoding complete.")

            for name, store in stores_to_query:
                logger.info(f"MiniLM vector search: Querying FAISS index '{name}'")
                distances, _, metadata_items = store.search(query_vector, k=k)
                logger.info(f"MiniLM vector search: FAISS index '{name}' returned {len(metadata_items)} results.")
                for i, item in enumerate(metadata_items):
                    all_results.append({
                        "id": item['id'], "label": item['label'], "score": float(distances[i]),
                        "source": "minilm", "source_ontology": name,
                    })
        except Exception as e:
            logger.error(f"Error during MiniLM vector search for '{query_string}': {e}", exc_info=True)

        all_results.sort(key=lambda x: x['score'])
        if all_results:
            top_ids = [r['id'] for r in all_results[:10]]
            top_scores = [f"{r['score']:.4f}" for r in all_results[:10]]
            logger.debug(f"[RETRIEVER_MINILM_RESULTS] query='{query_string}' | top_ids={top_ids} | top_scores={top_scores}")
        logger.info(f"Finished MiniLM vector search for query: '{query_string}'")
        return all_results

    def _vector_search_sapbert(self, query_string: str, k: int = None, target_ontologies: Optional[List[str]] = None):
        k = k if k is not None else config.DEFAULT_K_SAPBERT
        """Performs vector search using SapBERT biomedical embeddings."""
        logger.info(f"Starting SapBERT vector search for query: '{query_string}'")
        all_results = []
        if not query_string:
            return all_results

        stores_to_query = self._get_stores_to_query(self.faiss_stores_sapbert, target_ontologies)
        # Filter out None stores (where SapBERT index doesn't exist)
        stores_to_query = [(name, store) for name, store in stores_to_query if store is not None]

        if not stores_to_query:
            logger.warning("SapBERT vector search: No valid target ontologies with SapBERT indexes found.")
            return []

        try:
            # SapBERT (PubMedBERT tokenizer) can map uppercase gene symbols
            # to [UNK] (e.g., STAT1, BMP7). Use normalized form when available.
            normalized_query = normalize_biomedical_text(query_string)
            sapbert_query = normalized_query or query_string
            if sapbert_query != query_string:
                logger.info(
                    f"SapBERT vector search: using normalized query '{sapbert_query}' "
                    f"(original: '{query_string}')"
                )
            else:
                logger.info(f"SapBERT vector search: using raw query '{query_string}'")

            query_vector = self.embedding_model_sapbert.encode([sapbert_query], convert_to_numpy=True)
            logger.info("SapBERT vector search: Query encoding complete.")

            for name, store in stores_to_query:
                logger.info(f"SapBERT vector search: Querying FAISS index '{name}'")
                distances, _, metadata_items = store.search(query_vector, k=k)
                logger.info(f"SapBERT vector search: FAISS index '{name}' returned {len(metadata_items)} results.")
                for i, item in enumerate(metadata_items):
                    all_results.append({
                        "id": item['id'],
                        "label": item['label'],
                        "score": float(distances[i]),
                        "source": "sapbert",
                        "source_ontology": name,
                        "sapbert_query_variant": sapbert_query,
                    })
        except Exception as e:
            logger.error(f"Error during SapBERT vector search for '{query_string}': {e}", exc_info=True)

        all_results.sort(key=lambda x: x['score'])
        if all_results:
            top_ids = [r['id'] for r in all_results[:10]]
            top_scores = [f"{r['score']:.4f}" for r in all_results[:10]]
            logger.debug(f"[RETRIEVER_SAPBERT_RESULTS] query='{query_string}' | top_ids={top_ids} | top_scores={top_scores}")
        logger.info(f"Finished SapBERT vector search for query: '{query_string}'")
        return all_results

    def search(self, query_string: str, lexical_limit: int = None, minilm_k: int = None, sapbert_k: int = None, target_ontologies: Optional[List[str]] = None, trace_id: str = None):
        lexical_limit = lexical_limit if lexical_limit is not None else config.DEFAULT_K_LEXICAL
        minilm_k = minilm_k if minilm_k is not None else config.DEFAULT_K_MINILM
        sapbert_k = sapbert_k if sapbert_k is not None else config.DEFAULT_K_SAPBERT
        """
        Performs tri-hybrid search with 3 sources and deterministic deduplication.

        Retrieval sources:
        - Lexical (Whoosh BM25): Exact/fuzzy text matching
        - MiniLM vector: General semantic similarity
        - SapBERT vector: Biomedical domain-specific semantic similarity

        Merge strategy:
        - Deduplicate by (ontology, ID)
        - Track all contributing sources per entity
        - Rank final list with Reciprocal Rank Fusion (RRF) across sources
        """
        logger.info(f"Retriever.search initiated for query: '{query_string}'")

        # Execute all 3 searches
        lexical_results = self._lexical_search(query_string, limit=lexical_limit, target_ontologies=target_ontologies)
        minilm_results = self._vector_search_minilm(query_string, k=minilm_k, target_ontologies=target_ontologies)
        sapbert_results = self._vector_search_sapbert(query_string, k=sapbert_k, target_ontologies=target_ontologies)

        # --- 3-WAY DEDUPLICATION ---
        seen: Dict[tuple, Dict[str, Any]] = {}
        source_tracking: Dict[tuple, set] = {}

        # Process lexical first (higher priority as representative payload)
        for cand in lexical_results:
            key = (cand.get("source_ontology", ""), cand.get("id", ""))
            if key not in seen:
                seen[key] = dict(cand)
                source_tracking[key] = {"lexical"}
            else:
                source_tracking[key].add("lexical")
                if cand.get("score", 0) > seen[key].get("score", 0):
                    seen[key] = dict(cand)

        # Priority 2: SapBERT representative payload if not already lexical
        for cand in sapbert_results:
            key = (cand.get("source_ontology", ""), cand.get("id", ""))
            if key not in seen:
                seen[key] = dict(cand)
                source_tracking[key] = {"sapbert"}
            else:
                source_tracking[key].add("sapbert")
                if seen[key].get("source") == "sapbert":
                    if cand.get("score", float("inf")) < seen[key].get("score", float("inf")):
                        seen[key] = dict(cand)

        # Priority 3: MiniLM representative payload if not already lexical/sapbert
        for cand in minilm_results:
            key = (cand.get("source_ontology", ""), cand.get("id", ""))
            if key not in seen:
                seen[key] = dict(cand)
                source_tracking[key] = {"minilm"}
            else:
                source_tracking[key].add("minilm")
                if seen[key].get("source") == "minilm":
                    if cand.get("score", float("inf")) < seen[key].get("score", float("inf")):
                        seen[key] = dict(cand)

        lexical_ids = {(c.get("source_ontology", ""), c.get("id", "")) for c in lexical_results}
        minilm_ids = {(c.get("source_ontology", ""), c.get("id", "")) for c in minilm_results}
        sapbert_ids = {(c.get("source_ontology", ""), c.get("id", "")) for c in sapbert_results}

        def _first_rank_map(results: List[Dict[str, Any]]) -> Dict[tuple, int]:
            rank_map: Dict[tuple, int] = {}
            for rank, cand in enumerate(results, start=1):
                key = (cand.get("source_ontology", ""), cand.get("id", ""))
                if key not in rank_map:
                    rank_map[key] = rank
            return rank_map

        lexical_rank = _first_rank_map(lexical_results)
        minilm_rank = _first_rank_map(minilm_results)
        sapbert_rank = _first_rank_map(sapbert_results)

        # Reciprocal Rank Fusion score
        rrf_k = getattr(config, "RRF_K", 60)
        all_keys = set(lexical_rank) | set(minilm_rank) | set(sapbert_rank)
        fused_scores: Dict[tuple, float] = {}
        for key in all_keys:
            score = 0.0
            if key in lexical_rank:
                score += 1.0 / (rrf_k + lexical_rank[key])
            if key in minilm_rank:
                score += 1.0 / (rrf_k + minilm_rank[key])
            if key in sapbert_rank:
                score += 1.0 / (rrf_k + sapbert_rank[key])
            fused_scores[key] = score

        # Calculate overlaps
        overlap_lex_sapbert = len(lexical_ids & sapbert_ids)
        overlap_lex_minilm = len(lexical_ids & minilm_ids)
        overlap_sapbert_minilm = len(sapbert_ids & minilm_ids)
        overlap_all_three = len(lexical_ids & sapbert_ids & minilm_ids)

        merged_candidates = []
        for key, cand in seen.items():
            candidate = dict(cand)
            candidate["retrieval_sources"] = sorted(list(source_tracking[key]))
            candidate["fusion_score"] = fused_scores.get(key, 0.0)
            merged_candidates.append(candidate)

        # Sort by fused score first; use source/score as deterministic tie-breakers.
        merged_candidates.sort(key=lambda x: (
            -x.get("fusion_score", 0.0),
            0 if x.get("source") == "lexical" else (1 if x.get("source") == "sapbert" else 2),
            -x.get("score", 0) if x.get("source") == "lexical" else x.get("score", 0),
        ))

        merge_stats = {
            "lexical_k": len(lexical_results),
            "minilm_k": len(minilm_results),
            "sapbert_k": len(sapbert_results),
            "overlap_lex_sapbert": overlap_lex_sapbert,
            "overlap_lex_minilm": overlap_lex_minilm,
            "overlap_sapbert_minilm": overlap_sapbert_minilm,
            "overlap_all_three": overlap_all_three,
            "unique_after_dedupe": len(merged_candidates),
            "final_k": len(merged_candidates),
            "rrf_k": rrf_k,
        }
        logger.debug(f"[RETRIEVER_MERGE_STATS] query='{query_string}' | {json.dumps(merge_stats)}")

        candidate_tuples = [
            {
                "ontology": c.get("source_ontology"),
                "id": c.get("id"),
                "label": c.get("label"),
                "score": round(c.get("score", 0), 4),
                "source": c.get("source"),
                "fusion_score": round(c.get("fusion_score", 0.0), 6),
                "sources": c.get("retrieval_sources", []),
            }
            for c in merged_candidates[:20]
        ]
        logger.debug(f"[RETRIEVER_CANDIDATES_FINAL] query='{query_string}' | candidates={json.dumps(candidate_tuples, ensure_ascii=False)}")

        logger.info(f"Retriever.search finished for query: '{query_string}' | {merge_stats}")
        return {
            "query": query_string,
            "lexical_results": lexical_results,
            "minilm_results": minilm_results,
            "sapbert_results": sapbert_results,
            "merged_candidates": merged_candidates,
            "merge_stats": merge_stats,
        }

    async def search_async(self, query_string: str, lexical_limit: int = None, minilm_k: int = None, sapbert_k: int = None, target_ontologies: Optional[List[str]] = None, trace_id: str = None):
        """
        Async wrapper for search that runs blocking operations in a thread pool.
        This prevents blocking the event loop during CPU-bound embedding and index operations.
        """
        import asyncio
        return await asyncio.to_thread(self.search, query_string, lexical_limit, minilm_k, sapbert_k, target_ontologies, trace_id)

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
        logger.info("===== BASIC RETRIEVAL TEST =====")
        results = retriever.search("cheese", lexical_limit=2, minilm_k=2, sapbert_k=2)
        logger.info(f"Search 'cheese' returned {len(results['merged_candidates'])} candidates")

        logger.info("===== ONTOLOGY-SPECIFIC TEST: FoodOn =====")
        # Only query the foodon index
        results_foodon = retriever.search("cheese", lexical_limit=2, minilm_k=2, sapbert_k=2, target_ontologies=["foodon"])
        logger.info(f"FoodOn-specific search returned {len(results_foodon['merged_candidates'])} candidates")

        print("\n\n" + "="*80)
        print("✅ 3. Searching for 'chemical entity' ONLY in the 'chebi' ontology")
        print("="*80)
        results_chebi = retriever.search("chemical entity", lexical_limit=2, minilm_k=2, sapbert_k=2, target_ontologies=["chebi"])
        print(json.dumps(results_chebi, indent=2))

        print("\n\n" + "="*80)
        print("❌ 4. Searching with an invalid ontology name (should be skipped gracefully)")
        print("="*80)
        results_invalid = retriever.search("cheese", lexical_limit=2, minilm_k=2, sapbert_k=2, target_ontologies=["non_existent_ontology"])
        print(json.dumps(results_invalid, indent=2))

    except FileNotFoundError as e:
        logger.error(f"\nERROR: A required file was not found: {e}. Please ensure you have run the data ingestion and indexing scripts first.")
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during example run: {e}", exc_info=True)
    finally:
        if retriever:
            retriever.close()
