#!/usr/bin/env bash
#
# Rebuild the whole Onto-RAG pipeline from raw ontology → evaluation
# Usage: ./scripts/rebuild_all.sh            # runs with defaults
#        ./scripts/rebuild_all.sh --skip-eval   # skip final recall test
#
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${ROOT_DIR}"

DATA_DIR="data"

echo "🧹  Cleaning old artefacts …"
rm -f "${DATA_DIR}"/faiss_index.bin  "${DATA_DIR}"/faiss_metadata.json
rm -rf "${DATA_DIR}/whoosh_index"

echo "①  Parsing ontology …"
python -m src.ingestion.parse_ontology

echo "②  Creating enriched documents …"
python -m src.ingestion.enrich_documents

echo "③  Embedding documents …"
python -m src.embeddings.embed_docs

echo "④  Building FAISS index …"
python -m src.vector_store.faiss_store

echo "⑤  Building Whoosh index …"
python -m src.ingestion.build_lexical_index

if [[ "${1-}" != "--skip-eval" ]]; then
  echo "⑥  Running evaluator …"
  python -m src.evaluation.evaluate_retriever_recall
fi

echo "✅  Pipeline finished."
