# src/main.py
import argparse
import json
import logging
import asyncio

# Get a logger instance for this module
logger = logging.getLogger(__name__)

from src.adapters.pipeline_factory import create_pipeline
from src.config import DEFAULT_K_LEXICAL, DEFAULT_K_VECTOR, PIPELINE
from src.utils.logging_config import setup_run_logging
from src.utils.token_tracker import token_tracker

async def main():
    parser = argparse.ArgumentParser(description="Run the Onto-RAG pipeline with LLM selection.")
    parser.add_argument("query", type=str, help="The entity or text to search for (e.g., 'raw milk').")
    parser.add_argument("--lexical_k", type=int, default=DEFAULT_K_LEXICAL, help=f"Initial candidates from lexical search (default: {DEFAULT_K_LEXICAL}).")
    parser.add_argument("--vector_k", type=int, default=DEFAULT_K_VECTOR, help=f"Initial candidates from vector search (default: {DEFAULT_K_VECTOR}).")
    parser.add_argument("--top_n_rerank", type=int, default=100, help="Number of candidates to rerank and pass to the LLM (default: 10).")
    parser.add_argument("--show_candidates", action="store_true", help="Show the list of candidates provided to the LLM.")
    args = parser.parse_args()

    setup_run_logging(args.query)

    logger.info(f"Starting pipeline run for query: '{args.query}' with pipeline: '{PIPELINE}'")
    
    pipeline = None
    try:
        pipeline = create_pipeline(PIPELINE)
        
        result_tuple = await pipeline.run(
            query=args.query,
            lexical_k=args.lexical_k,
            vector_k=args.vector_k
        )
        
        results: list[dict] = []
        candidates = []
        if not result_tuple:
            logger.warning("Pipeline did not return a result.")
        else:
            results, candidates = result_tuple

        # Use print() for the final, user-facing output. This is the script's "result".
        print("\n--- Final LLM Selection ---")
        if not results:
            print("Could not determine a matching ontology term.")
        else:
            print(f"Query: '{args.query}'")
            print("---------------------------")
            for idx, final_result in enumerate(results[:3], start=1):
                print(f"Rank {idx}")
                print(f"  Term ID:      {final_result.get('id', 'N/A')}")
                print(f"  Label:        {final_result.get('label', 'N/A')}")
                print(f"  Confidence:   {final_result.get('confidence_score', 0.0):.1%}")
                print(f"  Selector conf:{final_result.get('selector_confidence', 'n/a')}")
                print(f"  Definition:   {final_result.get('definition', 'N/A')}")
                print(f"  Synonyms:     {'; '.join(final_result.get('synonyms', [])) or 'None'}")
                print(f"  Selector expl:{final_result.get('selector_explanation', 'No explanation provided.')}")
                print(f"  Scorer expl:  {final_result.get('scorer_explanation', 'No explanation provided.')}")
                print("")
        print("---------------------------\n")

        if args.show_candidates and candidates:
            # Also use print() here as this is user-requested output via a command-line flag.
            print(f"--- Top {len(candidates)} Candidates Provided to LLM ---")
            chosen_ids = {r.get('id') for r in results}
            
            for i, candidate in enumerate(candidates):
                details = pipeline.retriever.get_term_details(candidate.get('id'))
                if not details: continue

                marker = "⭐️" if details.get('id') in chosen_ids else "  "
                rerank_score = candidate.get('rerank_score')
                score_str = f"(Score: {rerank_score:.4f})" if rerank_score is not None else ""

                print(f"{i+1}. {marker} {details.get('label', 'N/A')} `{details.get('id', 'N/A')}` {score_str}")
                definition = details.get('definition')
                if definition:
                    print(f"       Def: {definition[:150]}...")
                else:
                    print(f"       Def: No definition available.")

                if details.get('synonyms'):
                    print(f"       Syns: {'; '.join(details.get('synonyms', []))}")
                print("-" * 20)
            print("-------------------------------------------\n")
        elif args.show_candidates:
            print("--- No Candidates to Display ---")

    except Exception as e:
        # Use the logger for exceptions. exc_info=True includes the traceback.
        logger.error(f"An unhandled error occurred during the pipeline execution: {e}", exc_info=True)
    
    finally:
        if pipeline:
            logger.info("Closing pipeline resources.")
            pipeline.close()
            
        print(token_tracker.report_usage())

if __name__ == "__main__":
    asyncio.run(main())
