# scripts/format_evaluation_results.py

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src import config
except ImportError:
    print("Error: Could not import from 'src'. Make sure the script is run from the project root"
          " or the project structure is correct.")
    # Provide fallback paths if config fails to import
    class config:
        ONTOLOGY_DUMP_JSON = project_root / "data" / "ontology_dump.json"

def load_json_data(file_path: Path) -> Any:
    """Loads data from a JSON file."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_enriched_details(curie: str, ontology_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves detailed information for a given CURIE from the ontology dump.
    
    Args:
        curie: The CURIE to look up (e.g., "FOODON:03316347").
        ontology_data: The loaded ontology_dump.json data.

    Returns:
        A dictionary with enriched details.
    """
    term_data = ontology_data.get(curie)
    if not term_data:
        return {
            "curie": curie,
            "label": "--- CURIE NOT FOUND ---",
            "definition": "N/A",
            "synonyms": []
        }
    
    return {
        "curie": curie,
        "label": term_data.get("label"),
        "definition": term_data.get("definition"),
        "synonyms": term_data.get("synonyms", [])
    }

def process_evaluation_file(
    eval_results: List[Dict[str, Any]],
    ontology_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Processes the evaluation results to create a human-readable version.
    """
    print("Enriching evaluation results...")
    enriched_output = []

    for item in eval_results:
        query = item.get("query")
        chosen_curie = item.get("chosen_curie")
        true_curies = item.get("true_curies", [])
        candidates_provided = item.get("candidates_provided", [])
        
        # Check if the chosen CURIE is in the list of true CURIEs
        is_correct = chosen_curie in true_curies

        # Enrich all relevant CURIEs
        chosen_details = get_enriched_details(chosen_curie, ontology_data)
        true_details = [get_enriched_details(tc, ontology_data) for tc in true_curies]
        candidates_details = [get_enriched_details(cc, ontology_data) for cc in candidates_provided]

        enriched_item = {
            "query": query,
            "is_correct": is_correct,
            "selector_explanation": item.get("selector_explanation"),
            "scorer_explanation": item.get("scorer_explanation"),
            "chosen_term": chosen_details,
            "ground_truth_terms": true_details,
            "candidate_terms_provided": candidates_details
        }
        enriched_output.append(enriched_item)
        
    return enriched_output

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Enrich LLM evaluation results with ontology details for human review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root / "data" / "evaluation_results_CRAFT_CHEBI_gemini.json",
        help="Path to the raw evaluation results JSON file."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=project_root / "data" / "evaluation_results_CRAFT_CHEBI_gemini_readable.json",
        help="Path to save the enriched, human-readable JSON file."
    )
    parser.add_argument(
        "--ontology-dump",
        type=Path,
        default=Path("./data/ontology_dump_chebi.json"),
        help="Path to the ontology_dump.json file."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    ontology_data = load_json_data(args.ontology_dump)
    eval_results = load_json_data(args.input_file)

    # Process and enrich the data
    readable_results = process_evaluation_file(eval_results, ontology_data)

    # Save the new file
    print(f"Saving enriched results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(readable_results, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()