# src/main.py
import argparse
import json
import os
import sys

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Assuming you have an ollama_pipeline, but this works for gemini_pipeline too
from src.pipeline.ollama_pipeline import RAGPipeline 
from src.config import DEFAULT_K_LEXICAL, DEFAULT_K_VECTOR

def main():
    parser = argparse.ArgumentParser(description="Run the Onto-RAG pipeline with LLM selection.")
    parser.add_argument("query", type=str, help="The entity or text to search for (e.g., 'raw milk').")
    parser.add_argument("--lexical_k", type=int, default=DEFAULT_K_LEXICAL, help=f"Initial candidates from lexical search (default: {DEFAULT_K_LEXICAL}).")
    parser.add_argument("--vector_k", type=int, default=DEFAULT_K_VECTOR, help=f"Initial candidates from vector search (default: {DEFAULT_K_VECTOR}).")
    parser.add_argument("--top_n_rerank", type=int, default=100, help="Number of candidates to rerank and pass to the LLM (default: 10).")
    parser.add_argument("--show_candidates", action="store_true", help="Show the list of candidates provided to the LLM.")
    args = parser.parse_args()

    pipeline = None
    try:
        pipeline = RAGPipeline()
        
        # --- MODIFIED: Unpack the tuple returned by pipeline.run ---
        result_tuple = pipeline.run(
            query=args.query,
            lexical_k=args.lexical_k,
            vector_k=args.vector_k,
            rerank_top_n=args.top_n_rerank
        )
        
        # Handle case where pipeline returns None
        if not result_tuple:
            final_result, candidates = None, []
        else:
            final_result, candidates = result_tuple

        # --- Print the final selection (no changes here) ---
        print("\n--- Final LLM Selection ---")
        if not final_result:
            print("Could not determine a matching ontology term.")
        else:
            print(f"Query: '{args.query}'")
            print("---------------------------")
            print(f"Chosen Term ID: {final_result.get('id', 'N/A')}")
            print(f"Label:          {final_result.get('label', 'N/A')}")
            print(f"Confidence:     {final_result.get('confidence_score', 0.0):.1%}")
            print(f"Definition:     {final_result.get('definition', 'N/A')}")
            print(f"Synonyms:       {'; '.join(final_result.get('synonyms', [])) or 'None'}")
            print("\nLLM Explanation:")
            print(f"  > {final_result.get('explanation', 'No explanation provided.')}")
        print("---------------------------\n")

        # --- NEW: Print the candidates if requested ---
        if args.show_candidates and candidates:
            print(f"--- Top {len(candidates)} Candidates Provided to LLM ---")
            chosen_id = final_result.get('id') if final_result else None
            
            for i, candidate in enumerate(candidates):
                # Fetch full details for printing
                details = pipeline.retriever.get_term_details(candidate.get('id'))
                if not details: continue

                marker = "⭐️" if details.get('id') == chosen_id else "  "
                rerank_score = candidate.get('rerank_score')
                score_str = f"(Score: {rerank_score:.4f})" if rerank_score is not None else ""

                print(f"{i+1}. {marker} {details.get('label', 'N/A')} `{details.get('id', 'N/A')}` {score_str}")
                definition = details.get('definition')
                if definition:
                    print(f"       Def: {definition[:150]}...")  # Print first 150 chars of definition
                else:
                    print(f"       Def: No definition available.")

                if details.get('synonyms'):
                    print(f"       Syns: {'; '.join(details.get('synonyms', []))}")
                print("-" * 20)
            print("-------------------------------------------\n")
        elif args.show_candidates:
            print("--- No Candidates to Display ---")


    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.close()

if __name__ == "__main__":
    main()