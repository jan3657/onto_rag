import json
import os
import logging
from typing import Dict, Any, List, Optional

# --- Start of corrected import block ---
import sys
# Calculate the project root directory based on the script's location
# For .../onto_rag/src/ingestion/enrich_documents.py, _PROJECT_ROOT becomes .../onto_rag
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to sys.path if it's not already there
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT) # Insert at the beginning

try:
    from src import config
    # from src.utils.ontology_utils import curie_to_uri # This specific utility is not used in this script,
                                                      # but this is how you'd import it if needed.
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR: Could not import project modules. Exception: {e}")
    print(f"This script expects to be run in a way that the 'src' package is discoverable.")
    print(f"Attempted to add project root '{_PROJECT_ROOT}' to sys.path.")
    print(f"Current sys.path: {sys.path}")
    print("Please ensure you are running this script from the project's root directory ('onto_rag/'), for example:")
    print("  python src/ingestion/enrich_documents.py")
    print("Also ensure that 'src/__init__.py' and 'src/utils/__init__.py' (if using utils) exist.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_label_for_curie(curie: str, ontology_data: Dict[str, Dict[str, Any]], default_value: Optional[str] = None) -> Optional[str]:
    """
    Retrieves the label for a given CURIE from the ontology data.
    Args:
        curie: The CURIE string (e.g., "FOODON:00001234").
        ontology_data: The loaded ontology data dictionary.
        default_value: Value to return if CURIE not found or has no label. Defaults to the CURIE itself.
    Returns:
        The label string, or the default_value.
    """
    if default_value is None:
        default_value = curie # Fallback to CURIE if no specific default

    term_info = ontology_data.get(curie)
    if term_info and term_info.get("label"):
        return term_info["label"]
    return default_value

def get_relation_name(relation_curie: str) -> str:
    """
    Gets a human-readable name for a relation CURIE using RELATION_CONFIG.
    """
    # Exact match
    if relation_curie in config.RELATION_CONFIG:
        return config.RELATION_CONFIG[relation_curie]["label"]
    
    # Check for generic FoodON prefix if specific one not found
    generic_foodon_prefix = "obo:FOODON_"
    if relation_curie.startswith(generic_foodon_prefix) and generic_foodon_prefix in config.RELATION_CONFIG:
         # Attempt to make it slightly more readable if it's like "obo:FOODON_0000XXXX"
        relation_suffix = relation_curie.split('_')[-1]
        return f"FoodON relation {relation_suffix}" # or config.RELATION_CONFIG[generic_foodon_prefix]["label"]

    # Fallback for other OBO relations
    if relation_curie.startswith("obo:"):
        name_part = relation_curie.split(':')[-1].replace("_", " ")
        return name_part

    return relation_curie # Fallback to the CURIE itself

def create_enriched_documents(ontology_data_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Creates enriched text documents for each ontology entry.
    Args:
        ontology_data_path: Path to the ontology_dump.json file.
        output_path: Path to save the enriched_documents.json file.
    Returns:
        A list of enriched document dictionaries.
    """
    logging.info(f"Loading ontology data from {ontology_data_path}...")
    try:
        with open(ontology_data_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Ontology data file not found at {ontology_data_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {ontology_data_path}")
        return []

    logging.info(f"Successfully loaded {len(ontology_data)} terms.")

    enriched_docs = []

    for term_curie, term_data in ontology_data.items():
        doc_parts = []

        # 1. Label
        label = term_data.get("label")
        if not label:
            logging.warning(f"Term {term_curie} has no label. Skipping for enriched document (or using CURIE as label).")
            # Consider if we want to enrich docs for terms without labels. For now, let's use CURIE if no label.
            label = term_curie 
        
        doc_parts.append(f"{label}.")

        # 2. Definition
        definition = term_data.get("definition")
        if definition:
            doc_parts.append(f"{definition}.")

        # 3. Synonyms
        synonyms = term_data.get("synonyms")
        if synonyms:
            synonyms_text = "; ".join(synonyms)
            doc_parts.append(f"Also known as: {synonyms_text}.")

        # 4. Parents (direct subclasses)
        parent_curies = term_data.get("parents", [])
        if parent_curies:
            parent_labels = [get_label_for_curie(p_curie, ontology_data) for p_curie in parent_curies]
            parent_labels_filtered = [l for l in parent_labels if l] # Filter out None if get_label_for_curie returns None
            if parent_labels_filtered:
                if len(parent_labels_filtered) == 1:
                    doc_parts.append(f"Is a type of: {parent_labels_filtered[0]}.")
                else:
                    doc_parts.append(f"Is a type of: {'; '.join(parent_labels_filtered)}.")
        
        # 5. Relations (including facets expressed as object properties)
        relations = term_data.get("relations", {})
        relation_texts = []
        for rel_curie, target_curies_list in relations.items():
            rel_name = get_relation_name(rel_curie)
            target_labels = [get_label_for_curie(t_curie, ontology_data) for t_curie in target_curies_list]
            target_labels_filtered = [l for l in target_labels if l]
            if target_labels_filtered:
                relation_texts.append(f"{rel_name}: {', '.join(target_labels_filtered)}")
        
        if relation_texts:
            doc_parts.append("Key characteristics include: " + "; ".join(relation_texts) + ".")

        # Combine all parts into a single text
        enriched_text = " ".join(doc_parts).replace("..", ".").strip() # Clean up potential double periods

        enriched_docs.append({
            "id": term_curie,
            "label": label if label != term_curie else term_data.get("label", term_curie), # Store original label if available
            "text": enriched_text
        })

    logging.info(f"Created {len(enriched_docs)} enriched documents.")

    logging.info(f"Saving enriched documents to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_docs, f, indent=2)
        logging.info("Successfully saved enriched documents.")
    except IOError:
        logging.error(f"Error: Could not write enriched documents to {output_path}")

    return enriched_docs

def main():
    """Main function to create enriched documents."""
    # Ensure data directory exists (though config.py should handle it)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    enriched_documents = create_enriched_documents(
        ontology_data_path=config.ONTOLOGY_DUMP_JSON,
        output_path=config.ENRICHED_DOCUMENTS_FILE
    )

    if enriched_documents:
        logging.info(f"Processed {len(enriched_documents)} documents.")
        # Optionally print a sample
        if len(enriched_documents) > 0:
            logging.info("Sample enriched document:")
            logging.info(json.dumps(enriched_documents[0], indent=2))
    else:
        logging.warning("No enriched documents were created.")

if __name__ == "__main__":
    main()