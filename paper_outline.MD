# OntoRAG–Food: Ontology‑Aware RAG for Food Entity Normalization

Short title: OntoRAG–Food
Target venue: BIG FOOD, NUTRITION AND SUSTAINABLE DEVELOPMENT DATA MANAGEMENT AND ANALYSIS
Scope: Food/Nutrition/Safety/Sustainability only (currently supports FoodOn only). CHEBI referenced only for food ingredients/additives (out-of-scope for this paper).

## Abstract (150–200 words)
Inconsistent product and menu labels break nutrition analytics, safety reporting, and monitoring: synonyms, branding, and preparation variants obscure the underlying food entity. OntoRAG‑Food addresses this by mapping mentions to standardized FoodOn identifiers, enabling reliable joins, dashboards, and alerts. The system builds enriched descriptions from ontology labels, definitions, synonyms, and relations; retrieves top‑k lexical and vector candidates; applies an LLM selector to choose a single term; and uses a confidence assessor with a synonym‑expansion loop when confidence is low. Evaluated on a FoodOn‑labeled cafeteria dataset, OntoRAG‑Food achieves Acc@1 of 0.63–0.69 while supporting ingredient‑list standardization for downstream analysis. We provide scripts and artifacts for full reproducibility. Technical details (models, index types, parameters) are documented in the Appendix; the main contribution is a practical, ontology‑aware RAG pipeline that improves food entity normalization without task‑specific training. Future work includes calibrated confidence and hierarchical evaluation for safety‑critical nutrition analytics.

## 1. Introduction
- Problem: Heterogeneous food terminology across products, menus, and reports; synonymy (“icing sugar” vs “powdered sugar”), function vs. substance (“acidulant” vs “citric acid”), and multilingual/orthographic variants complicate normalization.
- Need: Robust alignment to standardized food ontologies (FoodOn) to enable analytics, safety monitoring, and nutritional assessment.
- Contributions:
  - Ontology‑aware RAG pipeline with hybrid retrieval (Whoosh+FAISS) and LLM selector/scorer loop tuned for food entities.
  - Reproducible ingestion/enrichment/embedding/indexing scripts and evaluation protocol over a FoodOn‑labeled cafeteria dataset.
  - Practical ingredient normalization workflow for product data (id/label outputs for downstream analytics).
- Food use‑cases:
  - Product labeling alignment and ingredient normalization (standardized IDs for downstream analytics).
  - Cafeteria/menu normalization for nutrition analytics (FoodOn‑labeled mentions).
  - Safety/additive reporting via FoodOn relations (e.g., “has ingredient”).
  - Nutrient analytics by joining canonical FoodOn IDs with composition tables (pipeline returns id+label).

## 2. Background & Related Work (Food‑only)
- Food ontologies: FoodOn (current target).
- Ontology‑aware RAG vs. supervised entity linking in food: pipeline leverages ontology structure (labels/synonyms/defs/relations) for retrieval + LLM reasoning.
- Baselines (food‑specific framing): lexical‑only retrieval, vector‑only retrieval, small LLM selector without confidence loop.
- OUT‑OF‑SCOPE (remove/reframe): generic biomedical NER/EL; SNOMED CT unless explicitly tied to food; CRAFT Chebi evaluation as a standalone benchmark (CHEBI is used only as supporting chemical ingredients inside food workflows).

## 3. Method: OntoRAG‑Food Pipeline
3.1 Ingestion (OWL → JSON)
- Parse FoodOn to a lightweight JSON with labels, synonyms, definitions, parents/ancestors, and selected relations.
- TODO: Record ontology version IRI/date and dump timestamp for reproducibility (see Appendix for scripts).

3.2 Enrichment (document texts)
- Build concise enriched text for each term combining label/definition/synonyms/relations; output enriched JSON (see Appendix for scripts).

3.3 Embeddings
- Encode enriched texts with a sentence‑embedding model (details in Appendix).

3.4 Indexing
- Lexical index over label, synonyms, and definitions; vector index over embeddings (details in Appendix).

3.5 Hybrid Retrieval
- Combine lexical and vector candidates per query; retrieve top‑k from each source (parameters in Appendix).

3.6 Selector (LLM)
- Input: formatted candidate list with id/label/definition/synonyms.
- Output: JSON { chosen_id, explanation }.
- Food‑focused instructions guide selection (prompt details in Appendix). Current defaults should be updated to the food template (TODO).

3.7 Confidence Scorer (LLM) + Early Stopping
- Input: chosen term details + other candidates (context only).
- Output: JSON { confidence_score ∈ [0,1], explanation ≤30 words, suggested_alternatives ≤3 }.
- Early stopping uses a confidence threshold; when low, the scorer can suggest better alternatives (parameters and prompts in Appendix). Switch to the food‑focused prompt (TODO).

3.8 Synonym Generator (LLM) Loop
- Triggered if score < threshold and scorer provides no alternatives.
- Output: synonyms to expand query; loop bounded by MAX_PIPELINE_LOOPS=4.
- Generates additional query forms to improve recall when needed (prompt details in Appendix). Switch to the food‑focused prompt (TODO).

3.9 Optional Reranking (Planned Integration)
- Cross‑encoder reranking is available but not yet integrated; we plan to insert it before selection and ablate its impact (see Appendix).

3.10 Orchestration
- Orchestrates retrieval → selection → confidence → suggestions/synonyms with limited retries; supports multiple LLM providers (implementation details in Appendix).

## 4. Data
4.1 Ontologies
- FoodOn: current target. The processed snapshot used in our experiments contains 28,787 FoodOn entities. Enriched docs and embeddings match this count.
- Note: CHEBI assets exist in the repository for chemical‑ingredient exploration but are out‑of‑scope for this paper. OntoRAG‑Food currently targets FoodOn mappings.
- TODO: Include FoodOn version IRI/date and dump timestamp; log provenance (see Appendix for asset paths).

4.2 Datasets
- Cafeteria/menu dataset with FoodOn‑labeled mentions (≈948 entities) used for evaluation.
- A collection of product ingredient lists used to demonstrate ingredient standardization.
- OUT‑OF‑SCOPE (unless reframed to food): SNOMED CT resources.

## 5. Evaluation
- Metrics (implemented): Acc@1 (full pipeline).
- Metrics (planned): Acc@K, hierarchical distance, and calibration (ECE, Brier) from confidence scores (TODO).
- Protocol: top‑k lexical and vector retrieval, confidence‑based early stopping, limited retries; full set evaluation on FoodOn‑labeled mentions. Latency and cost proxies to be added (TODO). Technical details in Appendix.

## 6. Results (reproducible from evaluation logs)
- Acc@1 on the cafeteria FoodOn dataset ranges from ≈0.63 to ≈0.69 with the LLM selector, depending on run conditions; a smaller baseline model achieves ≈0.41–0.44.
- We will include a table with exact numbers, valid attempts, and run settings. TODO: Convert logs to structured CSV; add latency and cost proxy.

## 7. Analysis & Ablations
- Candidate K: vary lexical_k, vector_k.
- Embedding model: swap all‑MiniLM‑L6‑v2 with alternatives (TODO: add config toggles).
- Reranker: on/off once integrated.
- Selector/scorer prompts: food vs. chebi templates; minimal vs. detailed.
- Synonym loop: on/off; loop depth sensitivity.
- Target ontology: FoodOn‑only (current).

## 8. Limitations & Ethics (Food Domain)
- Coverage gaps and regional/cultural naming variability; brand/marketing language vs. substance identity.
- Licensing/provenance for ontologies and product datasets.
- LLM judge bias and lack of calibrated confidence; avoid over‑trust in safety‑critical contexts (allergens/additives).
- Generic mentions vs. specific preparations (salts/hydrates/states) remain challenging; enforce conservative no‑match when ambiguous.

## 9. Reproducibility & Artifacts
- Environment: requirements.txt; `.env` with `GEMINI_API_KEY`.
- Build pipeline:
  1) Parse OWL → JSON: `python src/ingestion/parse_ontology.py`
  2) Enrich documents: `python src/ingestion/enrich_documents.py`
  3) Embeddings: `python src/embeddings/embed_docs.py`
  4) Whoosh index: `python src/ingestion/build_lexical_index.py`
  5) FAISS index + metadata: `src/infrastructure/retrieval/faiss_store.py` auto‑loads existing files; build from embeddings if needed.
- Evaluate: `python src/evaluation/evaluate_pipeline.py` (writes evaluation_results_{pipeline}.json; logs in logs/).
- Minimal demo: `python src/main.py "cheese" --show_candidates` (uses configured indexes and PIPELINE).
- Ingredient standardization: `python src/map_off_products.py` (reads data/outputs/parsed_ingredients_output.json; writes standardized_ingredients_output.json).
- Seeds/temps: LLM calls deterministic JSON prompts; sampling disabled in HF settings; record run timestamps + model names in logs.

## 10. Conclusion & Impact
- OntoRAG‑Food normalizes noisy food entities to FoodOn at scale using ontology‑aware retrieval and LLM reasoning, supporting nutrition analytics and safety reporting.
- Future work: cross‑ontology alignment; calibrated confidence and hierarchical metrics; active learning for domain‑specific ambiguity; latency/cost optimizations.

## 11. Figures & Tables
- Fig 1: OntoRAG‑Food pipeline diagram (ingestion → enrichment → embedding → hybrid retrieval → selector → scorer → synonym loop).
- Table 1: Ontology & dataset stats (FoodOn entity counts; evaluation set size).
- Table 2: Mapping accuracy on cafeteria dataset (Acc@1 with valid attempts; baseline comparison).
- Table 3: Ablations (top‑k, reranker on/off, prompts, synonym loop) — export run logs to CSV (TODO).
- Fig 2: Reliability diagram (ECE/Brier) using scorer confidence (TODO).

## 12. Appendix — Missing Evidence / TODOs
- Switch prompts in src/config.py to food templates: `final_selection_ingredients.tpl`, `confidence_assessment*.tpl`, `synonym_generation.tpl`.
- Integrate reranker into pipeline before selection; ablate.
- Add ontology version/date + timestamp to dumps; record in paper.
- Add Acc@K, hierarchical distance, and calibration metrics; include plots.
- Export structured CSV of runs from logs (accuracy, valid attempts, token usage, K, threshold, loop depth).

## 13. Appendix — Key Implementation Pointers
- Config & parameters: src/config.py (embedding model, K, thresholds, PIPELINE, prompt paths; RELATION_CONFIG).
- Retrieval: src/infrastructure/retrieval/hybrid_retriever.py; FAISS store: src/infrastructure/retrieval/faiss_store.py.
- LLM components: selectors/scorers/synonyms under src/application/*; provider adapters in src/adapters/*.
- Evaluation scripts: src/evaluation/evaluate_pipeline.py; retriever recall scaffold: src/evaluation/evaluate_retriever_recall.py.
- Data assets: ontologies/*.owl; data/ontology_dump_foodon.json; data/enriched_documents_foodon.json; data/embeddings_foodon.json; data/whoosh_index_foodon; data/faiss_index_foodon.bin.
