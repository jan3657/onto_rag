#!/usr/bin/env bash
#
# Rebuilds all data and indexes for the Onto-RAG pipeline.
# Loops through all ontologies defined in src/config.py.
#
# Usage: ./scripts/rebuild_base.bash            # runs with defaults
#        ./scripts/rebuild_base.bash --skip-eval   # skip final recall test
#
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${ROOT_DIR}"

DATA_DIR="data"

echo "🧹  Cleaning old data and index artifacts..."
# Use wildcards to clean all ontology-specific files and directories
rm -f "${DATA_DIR}"/ontology_dump_*.json
rm -f "${DATA_DIR}"/enriched_documents_*.json
rm -f "${DATA_DIR}"/embeddings_*.json
rm -f "${DATA_DIR}"/faiss_index_*.bin
rm -f "${DATA_DIR}"/faiss_metadata_*.json
rm -rf "${DATA_DIR}"/whoosh_index_*
echo "✅  Cleanup complete."
echo

# Each script now handles looping internally based on src/config.py
echo "①  Parsing all configured ontologies..."
python -m src.ingestion.parse_ontology
echo

echo "②  Creating all enriched document files..."
python -m src.ingestion.enrich_documents
echo

echo "③  Embedding all enriched document sets..."
python -m src.embeddings.embed_docs
echo

echo "④  Building all FAISS vector stores..."
python -m src.vector_store.faiss_store
echo

echo "⑤  Building all Whoosh lexical indexes..."
python -m src.ingestion.build_lexical_index
echo

# The evaluation script may need updates to work with the new retriever
if [[ "${1-}" != "--skip-eval" ]]; then
  echo "⑥  Running evaluator (Note: may need updates for multi-ontology setup)..."
  python -m src.evaluation.evaluate_retriever_recall
fi

echo "✅  Pipeline rebuild finished successfully."