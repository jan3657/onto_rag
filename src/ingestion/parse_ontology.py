# src/ingestion/parse_ontology.py
import sys
import os
import logging # Import logging

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

import rdflib
from rdflib import Graph, Namespace, URIRef, RDFS, OWL, RDF
from typing import Dict, List, Any
import json
import traceback

# Now import using the 'src' package prefix
from src.config import (
    FOODON_PATH,                # Adjusted: Was ONTOLOGY_FILE
    ONTOLOGY_DUMP_JSON,         # Adjusted: Was ONTOLOGY_DUMP_PATH
    CURIE_PREFIX_MAP,           # Adjusted: Was NAMESPACE_MAP
    RELATION_CONFIG,            # New: For relation names
    TARGET_RELATIONS_CURIES,    # New: For relation URIs
    IAO_NS_STR,                 # Using string constants for Namespace definitions
    OBOINOWL_NS_STR,
    # Add other NS_STR if needed for local Namespace objects
)
# Assuming src.utils.ontology_utils is already correct and uses CURIE_PREFIX_MAP
from src.utils.ontology_utils import uri_to_curie, curie_to_uri

# --- Logging Setup ---
# Using basicConfig as src.utils.logging.get_logger is "to be developed"
# You can customize this further if needed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# Define commonly used namespaces (can still use these locally for convenience)
# It's safer to use the full URI string from config if available, or construct it.
IAO = Namespace(IAO_NS_STR)
OBOINOWL = Namespace(OBOINOWL_NS_STR)
# FOODON_BASE_URI = None # Find FOODON base URI from CURIE_PREFIX_MAP
# for base, prefix in CURIE_PREFIX_MAP.items():
#     if prefix == "FOODON":
#         FOODON_BASE_URI = base
#         break
# FOODON = Namespace(FOODON_BASE_URI) if FOODON_BASE_URI else None
# If FOODON Namespace object is not strictly needed for queries, direct URI construction is fine.

def load_ontology(path: str) -> rdflib.Graph:
    g = Graph()
    try:
        logger.info(f"Loading ontology from: {path}")
        try:
            g.parse(path, format="application/rdf+xml")
        except Exception as e_xml:
            logger.warning(f"Failed to parse as RDF/XML: {e_xml}. Trying Turtle...")
            try:
                g.parse(path, format="turtle")
            except Exception as e_ttl:
                logger.warning(f"Failed to parse as Turtle: {e_ttl}. Trying auto-detection...")
                g.parse(path)

        logger.info(f"Ontology loaded successfully. Contains {len(g)} triples.")
        return g
    except FileNotFoundError:
        logger.error(f"Error: Ontology file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing ontology file {path}: {e}")
        traceback.print_exc()
        raise

def get_ancestors(g: Graph, term_uri: URIRef, prefix_map: Dict[str, str], visited_uris: set = None) -> List[str]:
    if visited_uris is None:
        visited_uris = set()

    ancestor_curies = set()
    for parent_uri in g.objects(subject=term_uri, predicate=RDFS.subClassOf):
        if isinstance(parent_uri, URIRef) and parent_uri != OWL.Thing:
            if parent_uri not in visited_uris:
                visited_uris.add(parent_uri)
                # Pass the prefix_map explicitly
                parent_curie = uri_to_curie(parent_uri, prefix_map)
                if parent_curie and parent_curie != str(parent_uri):
                    ancestor_curies.add(parent_curie)
                    ancestor_curies.update(get_ancestors(g, parent_uri, prefix_map, visited_uris))
    return list(ancestor_curies)


def extract_labels_and_synonyms(g: Graph, prefix_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    data = {}
    relevant_predicates = [
        RDFS.label,
        OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
        OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym
    ]
    
    processed_subjects = set()

    for pred in relevant_predicates:
        for s_uri in g.subjects(predicate=pred):
            if not isinstance(s_uri, URIRef) or s_uri in processed_subjects:
                continue
            
            # Pass the prefix_map explicitly
            curie = uri_to_curie(s_uri, prefix_map)
            # Optional: Filter for specific prefixes if desired, e.g., only FOODON, IAO, OBO terms.
            # This was present in your original script; keeping it commented for now for broader extraction.
            # if not any(curie.startswith(p) for p in ["FOODON:", "IAO:", "RO:", "BFO:", "obo:", "CHEBI:"]): # Example
            #     # logger.debug(f"Skipping non-ontology CURIE for labels/syns: {curie} from {s_uri}")
            #     continue
            if not curie or curie == str(s_uri): # Skip if not converted to a CURIE effectively
                continue

            if curie not in data:
                data[curie] = {"label": None, "synonyms": []}

            # Label
            label_val = g.value(subject=s_uri, predicate=RDFS.label)
            if label_val and isinstance(label_val, rdflib.Literal):
                data[curie]["label"] = str(label_val)

            # Synonyms
            current_synonyms = []
            for syn_prop in [OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
                             OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym]:
                for syn_obj in g.objects(subject=s_uri, predicate=syn_prop):
                    if isinstance(syn_obj, rdflib.Literal):
                        current_synonyms.append(str(syn_obj))
            
            # Ensure synonyms list exists and extend uniquely
            if "synonyms" not in data[curie] or data[curie]["synonyms"] is None: # Handle if somehow becomes None
                data[curie]["synonyms"] = []
            for s in current_synonyms:
                if s not in data[curie]["synonyms"]:
                    data[curie]["synonyms"].append(s)

            processed_subjects.add(s_uri)
            
    final_data = {k: v for k, v in data.items() if v.get("label") or v.get("synonyms")}
    logger.info(f"Extracted labels and synonyms for {len(final_data)} terms.")
    return final_data


def extract_definitions(g: Graph, prefix_map: Dict[str, str]) -> Dict[str, str]:
    definitions = {}
    definition_prop_uri = IAO['0000115'] # IAO:0000115 is 'definition'
    for s_uri in g.subjects(predicate=definition_prop_uri):
        if not isinstance(s_uri, URIRef):
            continue
        
        # Pass the prefix_map explicitly
        curie = uri_to_curie(s_uri, prefix_map)
        # Optional: Filter for specific prefixes
        # if not any(curie.startswith(p) for p in ["FOODON:", "IAO:", "RO:", "BFO:", "obo:", "CHEBI:"]):
        #     # logger.debug(f"Skipping non-ontology CURIE for definitions: {curie} from {s_uri}")
        #     continue
        if not curie or curie == str(s_uri):
            continue
        
        def_obj = g.value(subject=s_uri, predicate=definition_prop_uri)
        if def_obj and isinstance(def_obj, rdflib.Literal):
            definitions[curie] = str(def_obj)
            
    logger.info(f"Extracted definitions for {len(definitions)} terms.")
    return definitions

def extract_hierarchy(g: Graph, prefix_map: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    hierarchy_data = {}
    all_terms_in_hierarchy = set()
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef): all_terms_in_hierarchy.add(s)
        if isinstance(o, URIRef): all_terms_in_hierarchy.add(o)
    
    for term_uri in all_terms_in_hierarchy:
        if term_uri == OWL.Thing:
            continue

        # Pass the prefix_map explicitly
        curie = uri_to_curie(term_uri, prefix_map)
        # Optional: Filter
        # if not any(curie.startswith(p) for p in ["FOODON:", "IAO:", "RO:", "BFO:", "obo:", "CHEBI:"]):
        #     # logger.debug(f"Skipping non-ontology CURIE for hierarchy: {curie} from {term_uri}")
        #     continue
        if not curie or curie == str(term_uri):
            continue

        direct_parent_curies = []
        for parent_uri in g.objects(subject=term_uri, predicate=RDFS.subClassOf):
            if isinstance(parent_uri, URIRef) and parent_uri != OWL.Thing:
                # Pass the prefix_map explicitly
                parent_curie = uri_to_curie(parent_uri, prefix_map)
                if parent_curie and parent_curie != str(parent_uri):
                    direct_parent_curies.append(parent_curie)
        
        ancestor_curies = get_ancestors(g, term_uri, prefix_map, visited_uris=set())
        
        if direct_parent_curies or ancestor_curies:
            hierarchy_data[curie] = {
                "parents": list(set(direct_parent_curies)),
                "ancestors": list(set(ancestor_curies))
            }
            
    logger.info(f"Extracted hierarchy data for {len(hierarchy_data)} terms.")
    return hierarchy_data


# Adjusted to match outline: props_to_extract: Dict[str, str] (readable_name -> URI_string)
def extract_relations(g: Graph, props_to_extract: Dict[str, str], prefix_map: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    relations_data = {}
    
    for term_uri in g.subjects(unique=True):
        if not isinstance(term_uri, URIRef):
            continue

        # Pass the prefix_map explicitly
        curie = uri_to_curie(term_uri, prefix_map)
        # Optional: Filter
        # if not any(curie.startswith(p) for p in ["FOODON:", "IAO:", "RO:", "BFO:", "obo:", "CHEBI:"]):
        #     # logger.debug(f"Skipping non-ontology CURIE for relations: {curie} from {term_uri}")
        #     continue
        if not curie or curie == str(term_uri):
            continue

        term_specific_relations = {}
        for rel_readable_name, rel_uri_str in props_to_extract.items(): # rel_name is readable_name
            rel_uri = URIRef(rel_uri_str)
            target_curies = []
            for target_obj in g.objects(subject=term_uri, predicate=rel_uri):
                if isinstance(target_obj, URIRef):
                    # Pass the prefix_map explicitly
                    target_curie = uri_to_curie(target_obj, prefix_map)
                    if target_curie and target_curie != str(target_obj):
                        target_curies.append(target_curie)
            
            if target_curies:
                term_specific_relations[rel_readable_name] = list(set(target_curies))

        if term_specific_relations:
            relations_data[curie] = term_specific_relations
            
    logger.info(f"Extracted relations for {len(relations_data)} terms based on {len(props_to_extract)} specified properties.")
    return relations_data


def main():
    logger.info("Starting ontology parsing...")
    # Adjusted: Use ONTOLOGY_DUMP_JSON for output path
    data_dir = os.path.dirname(ONTOLOGY_DUMP_JSON)
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Adjusted: Use FOODON_PATH as input ontology file
        g = load_ontology(FOODON_PATH)

        # Prepare relation properties for extract_relations
        # extract_relations expects: Dict[readable_name, full_uri_string]
        # RELATION_CONFIG has: {curie_str: {"label": readable_name, "prefix": prefix_str}}
        # TARGET_RELATIONS_CURIES is List[curie_str]
        
        relation_properties_for_extraction = {}
        for rel_curie_str in TARGET_RELATIONS_CURIES:
            if rel_curie_str in RELATION_CONFIG:
                config_entry = RELATION_CONFIG[rel_curie_str]
                readable_name = config_entry.get("label", rel_curie_str) # Fallback to CURIE if no label
                
                # Convert relation CURIE to full URI using curie_to_uri and CURIE_PREFIX_MAP
                full_rel_uri = curie_to_uri(rel_curie_str, CURIE_PREFIX_MAP)
                if full_rel_uri:
                    relation_properties_for_extraction[readable_name] = str(full_rel_uri)
                else:
                    logger.warning(f"Could not convert relation CURIE {rel_curie_str} to URI. Skipping this relation.")
            else:
                logger.warning(f"Relation CURIE {rel_curie_str} from TARGET_RELATIONS_CURIES not found in RELATION_CONFIG.")
        
        logger.info(f"Prepared {len(relation_properties_for_extraction)} relation properties for extraction.")


        logger.info("\nExtracting data...")
        # Pass CURIE_PREFIX_MAP to all extraction functions
        labels_synonyms = extract_labels_and_synonyms(g, CURIE_PREFIX_MAP)
        definitions = extract_definitions(g, CURIE_PREFIX_MAP)
        hierarchy = extract_hierarchy(g, CURIE_PREFIX_MAP)
        relations = extract_relations(g, relation_properties_for_extraction, CURIE_PREFIX_MAP)

        logger.info("\nMerging extracted data...")
        merged_data = {}
        all_curies = set(labels_synonyms.keys()) | \
                     set(definitions.keys()) | \
                     set(hierarchy.keys()) | \
                     set(relations.keys())

        for curie_key in all_curies:
            # No need to filter again here if extraction functions already did or if we want all extracted CURIEs
            merged_data[curie_key] = {
                "label": labels_synonyms.get(curie_key, {}).get("label"),
                "synonyms": labels_synonyms.get(curie_key, {}).get("synonyms", []),
                "definition": definitions.get(curie_key),
                "parents": hierarchy.get(curie_key, {}).get("parents", []),
                "ancestors": hierarchy.get(curie_key, {}).get("ancestors", []),
                "relations": relations.get(curie_key, {})
            }
        
        final_merged_data = {}
        for curie_key, data_dict in merged_data.items():
            if any(data_dict.values()): # Check if any value in the dict is non-empty/non-None
                final_merged_data[curie_key] = data_dict

        logger.info(f"\nTotal merged entities with some data: {len(final_merged_data)}")

        # Adjusted: Use ONTOLOGY_DUMP_JSON for output
        logger.info(f"Writing merged data to {ONTOLOGY_DUMP_JSON}")
        with open(ONTOLOGY_DUMP_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_merged_data, f, indent=4, ensure_ascii=False)

        logger.info("Ontology parsing and data dump complete.")

    except FileNotFoundError:
        # Adjusted: Use FOODON_PATH in error message
        logger.error(f"Parsing aborted: Ontology file not found at {FOODON_PATH}")
        # traceback.print_exc() # Already handled by load_ontology
    except Exception as e:
        logger.error(f"An error occurred during parsing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()