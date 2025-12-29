# 🧬 OntoRAG

**Ontology-Augmented Retrieval for Named Entity Linking (NEL)**

OntoRAG is a hybrid retrieval-augmented generation (RAG) pipeline for linking free-text mentions (e.g., food ingredients, chemicals) to ontology terms. It combines lexical and semantic search with LLM-based selection and confidence scoring to achieve accurate entity linking.

---

## ✨ Features

- **Hybrid Retrieval** — Combines [Whoosh](https://whoosh.readthedocs.io/) lexical search and [FAISS](https://github.com/facebookresearch/faiss) vector search for comprehensive candidate retrieval
- **LLM-Powered Selection** — Uses structured prompts with Gemini or local models (Ollama, Hugging Face) to select the best ontology match
- **Confidence Scoring** — Evaluates match quality and suggests alternatives when confidence is low
- **Adaptive Retry Logic** — Automatically generates synonyms and retries when initial matches are uncertain
- **Multi-Ontology Support** — Pre-configured for [FoodOn](https://foodon.org/) with extensible architecture for ChEBI, MEDIC, and others
- **Evaluation Suite** — Built-in benchmarking against CRAFT and other gold-standard datasets

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query ──► HybridRetriever ──► Selector ──► ConfidenceScorer   │
│                 │                   │              │            │
│           ┌─────┴─────┐             │              ▼            │
│           │           │             │         Score < 0.6?      │
│        Whoosh      FAISS            │              │            │
│        (lexical)  (vector)          │              ▼            │
│                                     │     SynonymGenerator      │
│                                     │              │            │
│                                     └──────────────┤            │
│                                                    ▼            │
│                                               Best Match        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
onto_rag/
├── src/
│   ├── pipeline.py              # Main RAG pipeline orchestration
│   ├── config.py                # Configuration and model settings
│   ├── components/
│   │   ├── retriever.py         # HybridRetriever (Whoosh + FAISS)
│   │   ├── selector.py          # LLM-based candidate selection
│   │   ├── scorer.py            # Confidence scoring
│   │   ├── synonyms.py          # Query expansion via synonym generation
│   │   └── faiss_store.py       # FAISS vector store wrapper
│   ├── evaluation/              # Benchmark scripts (CRAFT, OpenFoodFacts)
│   └── utils/                   # Caching, logging, token tracking
├── prompts/                     # Jinja2 prompt templates
│   ├── strict_selection_minimal.tpl
│   ├── confidence_assessment3.tpl
│   └── synonym_generation.tpl
├── data/                        # Ontology dumps, indexes (gitignored)
├── ontologies/                  # OWL files (gitignored)
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd onto_rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here    # Optional
PINECONE_API_KEY=your_pinecone_api_key     # Optional
```

### 3. Prepare Data

Place your ontology files (e.g., `foodon.owl`) in the `ontologies/` directory, then run the indexing scripts to generate:

- Ontology JSON dumps in `data/`
- Whoosh lexical indexes
- FAISS vector indexes

### 4. Usage

```python
import asyncio
from src.pipeline import create_pipeline

async def main():
    pipeline = create_pipeline(provider="gemini")
    
    result, candidates = await pipeline.run(
        query="organic raw milk",
        context="Found in ingredient list of artisan cheese"
    )
    
    print(f"Matched: {result['label']} ({result['id']})")
    print(f"Confidence: {result['confidence_score']:.2f}")
    
    pipeline.close()

asyncio.run(main())
```

---

## ⚙️ Configuration Options

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `PIPELINE` | `"gemini"` | LLM provider: `gemini`, `ollama`, or `huggingface` |
| `CONFIDENCE_THRESHOLD` | `0.6` | Score below which synonyms are generated |
| `MAX_PIPELINE_LOOPS` | `4` | Maximum retry attempts per query |
| `DEFAULT_K_LEXICAL` | `15` | Number of lexical search results |
| `DEFAULT_K_VECTOR` | `15` | Number of vector search results |
| `MAX_CONCURRENT_REQUESTS` | `20` | Async concurrency limit for batch processing |

---

## 📊 Evaluation

Run benchmarks against standard datasets:

```bash
# Evaluate on CRAFT ChEBI
python -m src.evaluation.evaluate_craft_chebi --limit 100

# Evaluate on OpenFoodFacts
python -m src.evaluation.evaluate_on_off --limit 50

# Evaluate retriever recall
python -m src.evaluation.evaluate_retriever_recall
```

---

## 🔧 LLM Backends

### Gemini (Default)
Requires `GEMINI_API_KEY` in `.env`. Uses `gemini-2.5-flash-lite` by default.

### Ollama (Local)
```bash
# Install Ollama and pull a model
ollama pull llama3.1:8b

# Update config
PIPELINE = "ollama"
```

### Hugging Face (Local GPU)
Supports quantized models with `bitsandbytes`. Configured via `HF_SELECTOR_MODEL_ID` in config.

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

- [FoodOn](https://foodon.org/) — Food Ontology
- [ChEBI](https://www.ebi.ac.uk/chebi/) — Chemical Entities of Biological Interest
- [BioEL](https://github.com/bioel-project/bioel) — Biomedical Entity Linking framework
