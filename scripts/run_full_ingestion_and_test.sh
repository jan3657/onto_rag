#!/bin/bash
# Full ingestion and benchmark test script for OntoRAG
# Usage: ./scripts/run_full_ingestion_and_test.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OntoRAG Full Ingestion & Benchmark Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate environment
echo -e "${YELLOW}[1/5] Activating environment...${NC}"
source ~/venvs/onto_rag/bin/activate || {
    echo -e "${RED}Failed to activate environment${NC}"
    exit 1
}
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Ingestion 1: CTD Diseases
echo -e "${YELLOW}[2/5] Ingesting CTD Diseases ontology...${NC}"
python -m src.evaluation.evaluate_ctd_diseases --ingest 2>&1 | tee logs/ingest_ctd_diseases.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ CTD Diseases ingestion complete${NC}"
else
    echo -e "${RED}✗ CTD Diseases ingestion failed${NC}"
    exit 1
fi
echo ""

# Ingestion 2: CRAFT ChEBI
echo -e "${YELLOW}[3/5] Ingesting CRAFT ChEBI ontology...${NC}"
python -m src.evaluation.evaluate_craft_chebi --ingest 2>&1 | tee logs/ingest_craft_chebi.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ CRAFT ChEBI ingestion complete${NC}"
else
    echo -e "${RED}✗ CRAFT ChEBI ingestion failed${NC}"
    exit 1
fi
echo ""

# Ingestion 3: CAFETERIA FoodOn
echo -e "${YELLOW}[4/5] Ingesting CAFETERIA FoodOn ontology...${NC}"
python -m src.evaluation.evaluate_cafeteria_foodon --ingest 2>&1 | tee logs/ingest_cafeteria_foodon.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ CAFETERIA FoodOn ingestion complete${NC}"
else
    echo -e "${RED}✗ CAFETERIA FoodOn ingestion failed${NC}"
    exit 1
fi
echo ""

# Verify all FAISS files exist
echo -e "${YELLOW}Verifying ingestion outputs...${NC}"
DATASETS=("ncbi_gene" "ctd_diseases" "craft_chebi" "cafeteria_foodon")
ALL_GOOD=true

for ds in "${DATASETS[@]}"; do
    echo -e "  Checking ${ds}..."
    if [ -f "data/${ds}/faiss_index_sapbert.bin" ] && [ -f "data/${ds}/faiss_index_minilm.bin" ]; then
        echo -e "    ${GREEN}✓ SapBERT and MiniLM indexes found${NC}"
    else
        echo -e "    ${RED}✗ Missing FAISS indexes${NC}"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo -e "${RED}Some ingestions are incomplete. Please check logs.${NC}"
    exit 1
fi
echo ""

# Run benchmark test
echo -e "${YELLOW}[5/5] Running benchmark test (10 samples per dataset)...${NC}"
python -m src.evaluation.run_benchmark \
    --provider gemini \
    --limit 10 \
    --seed 42 \
    --no-cache 2>&1 | tee logs/benchmark_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark test complete${NC}"
else
    echo -e "${RED}✗ Benchmark test failed${NC}"
    exit 1
fi
echo ""

# Analyze results
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SOURCE ATTRIBUTION ANALYSIS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python3 << 'EOF'
import json
from pathlib import Path

try:
    results_dir = max(Path("data/benchmark_results").glob("202*"), key=lambda p: p.name)
    summary = json.load((results_dir / "summary.json").open())

    print("📊 Results Summary (10 samples per dataset)\n")
    print("="*70)

    for ds_name, ds_data in sorted(summary.get("datasets", {}).items()):
        sa = ds_data["metrics"].get("source_attribution", {})
        acc = ds_data["metrics"].get("accuracy", 0)

        print(f"\n🔹 {ds_name.upper()}")
        print(f"   Overall Accuracy: {acc*100:.1f}%")

        by_source = sa.get("by_source", {})
        if by_source:
            print("   Retrieval Sources:")
            for source in ["lexical_only", "minilm_only", "sapbert_only",
                           "lexical+minilm", "lexical+sapbert", "minilm+sapbert", "all_three"]:
                counts = by_source.get(source, {})
                total = counts.get("correct", 0) + counts.get("incorrect", 0)
                if total > 0:
                    correct = counts.get("correct", 0)
                    prec = sa.get("precision_by_source", {}).get(source, 0)
                    print(f"      • {source:20s}: {total:2d} predictions ({correct} correct, {prec:.0f}%)")

        print()

    print("="*70)
    print(f"\n📁 Full results: {results_dir}")
    print(f"   - summary.json")
    print(f"   - details.json")
    print(f"   - comparison.json")

except Exception as e:
    print(f"Error analyzing results: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ ALL TASKS COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  • Review results in data/benchmark_results/"
echo -e "  • Run full benchmark: python -m src.evaluation.run_benchmark --provider gemini"
echo -e "  • Check logs in logs/ directory"
echo ""
