# onto_rag

A Retrieval-Augmented Generation (RAG) pipeline for mapping noisy entity mentions onto curated ontology terms. The project couples traditional information retrieval with large language models (LLMs) and feedback-driven retries so that entity linking decisions are transparent, auditable, and tunable for multiple ontologies.

## Key Capabilities

* **Multi-ontology ingestion and indexing.** Data preparation scripts parse OWL ontologies, enrich the textual description of every term, build Whoosh lexical indexes, and construct FAISS vector stores for each configured ontology, allowing the runtime to serve FoodOn, ChEBI, and other resources side-by-side.
* **Hybrid retrieval pipeline.** The `HybridRetriever` fans out lexical (Whoosh) and vector (SentenceTransformers + FAISS) queries across all available ontologies and merges the results so downstream models receive a broad yet relevant candidate list for every user query.
* **LLM-based selection, scoring, and synonym expansion.** Provider-specific adapters reuse shared selector, confidence scorer, and synonym generator bases to format prompts, parse JSON responses, track token usage, and orchestrate retries with scorer feedback baked into every loop iteration.
* **Iterative reasoning loop with guardrails.** The core pipeline stops calling the scorer when selectors reject all candidates, feeds scorer explanations into subsequent retries, prioritises scorer-suggested alternatives, and only falls back to synonym generation when necessary.
* **Cross-encoder reranking.** An optional `LLMReranker` loads enriched documents and scores candidate/query pairs with a sentence-transformer cross encoder to tighten the shortlist before the LLM selector runs.
* **Caching and observability.** Utilities capture aggregate token usage, maintain persistent caches of high-confidence answers, and expose verbose execution traces via a Streamlit demo app for rapid inspection of every pipeline iteration.
* **Evaluation tooling.** End-to-end scoring scripts load gold-standard annotations, reuse the production pipeline asynchronously, and persist per-entity decisions to JSON for downstream analysis, while helper functions build ±100-character evidence windows around each mention.

## Architecture Overview

### Configuration and Ontology Assets
Centralised configuration maps ontology identifiers to their source OWL files, derived artefact locations, embedding models, loop limits, and default LLM models. Each ontology entry lists the dump, enrichment, lexical index, and FAISS paths that the runtime loads on start-up.

### Data Preparation Workflow
1. **Ontology parsing.** `parse_ontology.py` extracts labels, synonyms, definitions, hierarchy, and relation edges into JSON dumps organised by CURIE.
2. **Document enrichment.** `enrich_documents.py` composes natural-language descriptions that blend labels, definitions, synonyms, and relation summaries for downstream embedding and reranking.
3. **Embedding generation and indexing.** Subsequent scripts embed the enriched documents, write FAISS indexes/metadata, and build Whoosh lexical indexes for rapid lookup.
4. **Automation.** `scripts/rebuild_base.bash` orchestrates the full rebuild sequence so environments stay synchronised when ontology sources change.

### Retrieval Layer
`HybridRetriever` opens every configured Whoosh index and FAISS store once, then serves lexical and vector searches per ontology. Results record the source modality and ontology key so later stages can reconcile duplicates or inspect provenance.

### LLM Decision Loop
The base pipeline composes interchangeable components:

1. **Selection.** `BaseSelector` formats candidate lists with definitions and synonyms, tolerates malformed JSON, and surfaces the model’s explanation for UI/display.
2. **Confidence scoring.** `BaseConfidenceScorer` compares the selector’s choice against the original query and remaining candidates, parsing optional alternative suggestions to guide retries.
3. **Synonym generation.** `BaseSynonymGenerator` injects scorer feedback directly into new synonym prompts so each retry is better informed than the last.
4. **Loop controller.** `BaseRAGPipeline` coordinates retrieval, selection, scoring, scorer-guided retries, synonym fallbacks, and semaphore-guarded concurrency. It short-circuits scorer calls when selectors refuse all candidates, persists the best-so-far result, and enforces loop and confidence thresholds configured in `config.py`.

Adapters under `src/adapters` plug Gemini, Ollama, or Hugging Face clients into the shared interfaces so teams can swap model providers without altering pipeline logic.

### Reranking and Post-processing
`LLMReranker` optionally refines candidate orderings before the selector runs by pairing the query with enriched term documents and scoring them via a cross encoder, ensuring the LLM receives contextually rich evidence.

### Utilities, Caching, and Reporting
Supporting utilities capture per-model token consumption, emit consistent logging, and persist pipeline caches so repeat entities can bypass expensive LLM calls during evaluations and product ingestion runs. Long-form evaluation scripts reuse these caches, log loop behaviour, and serialise final outputs for downstream adjudication.

## Getting Started

1. **Install dependencies.** Use Python 3.10+ and install requirements with `pip install -r requirements.txt`.
2. **Configure environment variables.** Copy `.env.example` to `.env`, then provide API keys for your selected LLM providers (Gemini, Ollama, or Hugging Face).
3. **Prepare ontology assets.** Place source OWL files in `ontologies/`, then run the ingestion pipeline (e.g., `scripts/rebuild_base.bash`) to generate dumps, enriched documents, embeddings, and indexes before launching any applications.

## Running the Pipeline

* **Programmatic access.** `src/adapters/pipeline_factory.create_pipeline` wires together the retriever, selector, scorer, and synonym generator for a chosen provider. You can call `await pipeline.run(query, context)` in your own orchestration code, then `pipeline.close()` when finished.
* **Interactive demo.** `pipeline_live_app.py` exposes the full verbose loop through a Streamlit interface, showing every prompt, raw LLM response, scorer suggestion, and generated synonym across iterations.
* **Batch product mapping.** `src/map_off_products.py` demonstrates integrating the pipeline into production dataflows, standardising ingredient lists while sharing a high-confidence cache across asynchronous workers.

## Evaluation

* `src/evaluation/evaluate_craft_chebi.py` loads BioCreative-style annotations, builds ±100-character context windows, and executes the asynchronous pipeline with optional caching to compute accuracy and export predictions.
* Additional evaluation result snapshots in `evaluation_results_*.json` capture selector and scorer rationales for past experiments, helping with regression analysis and prompt tuning.

## Roadmap

### Model Improvements

- [x] Do not call the scorer if there are no good candidates
- [x] Integrate scorer recommendations into the synonym generator
- [ ] Merge the scorer and synonym generator into a unified model
- [ ] Optimize and minimize prompt templates for better efficiency
- [x] ±100-char context window
- [x] Add the scorer feedback to the loop for iterative improvement
- [ ] If confidence is still low switch to a stronger model

### Performance & Infrastructure

- [x] Implement caching mechanism for improved response times
- [ ] Experiment with different embedding models for better accuracy
- [ ] Add pre-flight exact match checking to avoid unnecessary processing

### Testing & Evaluation

- [ ] Complete evaluation runs on the Cafeteria dataset
- [ ] Validate system performance with comprehensive test cases
