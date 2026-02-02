# Error Atlas Task Board

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Scaffolding
- [x] Capture high-level analysis goals and visualization ideas
- [ ] Wire a reproducible target (`make error-atlas`) to run the whole pipeline

## Phase 1 — Data consolidation
- [x] Build unified per-query dataframe from all `data/benchmark_results/*/results.jsonl`
- [x] Persist dataframe to `data/analysis/errors.parquet` + summary `errors.csv`
- [x] Derive `error_type` labels (retrieval_miss, ranking_miss, no_prediction, system_error, other)
- [x] Enrich rows with run metadata (model, provider, timestamp, concurrent_requests, sample_size)
- [x] Add lightweight lexical features (lowercased query, tokens, length, has_digits/greek/hyphen)

## Phase 2 — Embeddings & features
- [x] Generate baseline text embeddings (e5-small) for query text (incorrect rows)
- [ ] Compute differential vectors (query−gold, query−pred, pred−gold) and scalar margins
- [ ] Cache embeddings under `data/embeddings/<encoder>/`

## Phase 3 — 2D projections & plots
- [x] Run PCA→t-SNE reduction (incorrect set); store coordinates per encoder
- [x] Render interactive scatter with layers (model, dataset, error_type, confidence)
- [ ] Add bipartite link overlay (query→gold vs query→pred) for failures
- [ ] Build confusion graph of gold/pred IDs and project via node2vec→UMAP

## Phase 4 — Diagnostics
- [ ] Compute neighborhood purity and “universal hard” counts across models
- [ ] Regress error likelihood on features; surface SHAP/feature importances
- [ ] Export HTML report/notebook with slices per dataset and timestamp

## Stretch
- [ ] Prototype supervised UMAP using `error_type` as label
- [ ] Try domain encoder (SapBERT/PubMedBERT) and compare projections
