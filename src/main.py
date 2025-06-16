# src/main.py
import argparse
import json
import os
import sys

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.ollama_pipeline import RAGPipeline
from src.config import DEFAULT_K_LEXICAL, DEFAULT_K_VECTOR

def main():
    parser = argparse.ArgumentParser(description="Run the Onto-RAG pipeline with LLM selection.")
    parser.add_argument("query", type=str, help="The entity or text to search for (e.g., 'raw milk').")
    parser.add_argument("--lexical_k", type=int, default=DEFAULT_K_LEXICAL, help=f"Initial candidates from lexical search (default: {DEFAULT_K_LEXICAL}).")
    parser.add_argument("--vector_k", type=int, default=DEFAULT_K_VECTOR, help=f"Initial candidates from vector search (default: {DEFAULT_K_VECTOR}).")
    parser.add_argument("--top_n_rerank", type=int, default=10, help="Number of candidates to rerank and pass to the LLM (default: 10).")
    args = parser.parse_args()

    pipeline = None
    try:
        pipeline = RAGPipeline()
        final_result = pipeline.run(
            query=args.query,
            lexical_k=args.lexical_k,
            vector_k=args.vector_k,
            rerank_top_n=args.top_n_rerank
        )

        print("\n--- Final LLM Selection ---")
        if not final_result:
            print("Could not determine a matching ontology term.")
        else:
            print(f"Query: '{args.query}'")
            print("---------------------------")
            print(f"Chosen Term ID: {final_result.get('id', 'N/A')}")
            print(f"Label:          {final_result.get('label', 'N/A')}")
            print(f"Definition:     {final_result.get('definition', 'N/A')}")
            print(f"Synonyms:       {'; '.join(final_result.get('synonyms', [])) or 'None'}")
            print("\nLLM Explanation:")
            print(f"  > {final_result.get('explanation', 'No explanation provided.')}")
        print("---------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution: {e}", file=sys.stderr)
    finally:
        if pipeline:
            pipeline.close()

if __name__ == "__main__":
    main()