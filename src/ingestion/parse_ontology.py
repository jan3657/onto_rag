# src/ingestion/parse_ontology.py
import sys
import os

# --- Add project root to sys.path ---
# This ensures that 'src' can be imported as a top-level package.
# The path added is the parent directory of 'src/', i.e., 'onto_rag/'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

import rdflib
from rdflib import Graph, Namespace, URIRef, RDFS, OWL, RDF
from typing import Dict, List, Any
import json
import traceback # For detailed error logging

# Now import using the 'src' package prefix
from src.config import ONTOLOGY_FILE, ONTOLOGY_DUMP_PATH, NAMESPACE_MAP, RELATION_PROPERTIES
from src.utils.ontology_utils import uri_to_curie, curie_to_uri # CORRECTED IMPORT


# Define commonly used namespaces (can still use these locally for convenience)
FOODON = Namespace(NAMESPACE_MAP["FOODON"])
IAO = Namespace(NAMESPACE_MAP["IAO"])
OBOINOWL = Namespace(NAMESPACE_MAP["OBOINOWL"])
# BFO and RO URIs should come directly from RELATION_PROPERTIES or NAMESPACE_MAP if needed
# BFO = Namespace(NAMESPACE_MAP["BFO"])
# RO = Namespace(NAMESPACE_MAP["RO"])


def load_ontology(path: str) -> rdflib.Graph:
    g = Graph()
    try:
        print(f"Loading ontology from: {path}")
        # RDFLib can often guess, but 'xml' is a common explicit choice for OWL/XML
        # For .owl files, it's typically rdf/xml or turtle.
        # Try common formats explicitly if auto-detection fails.
        try:
            g.parse(path, format="application/rdf+xml") # Common for .owl
        except Exception as e_xml:
            print(f"Failed to parse as RDF/XML: {e_xml}. Trying Turtle...")
            try:
                g.parse(path, format="turtle")
            except Exception as e_ttl:
                print(f"Failed to parse as Turtle: {e_ttl}. Trying auto-detection...")
                g.parse(path) # Fallback to auto-detection

        print(f"Ontology loaded successfully. Contains {len(g)} triples.")
        return g
    except FileNotFoundError:
        print(f"Error: Ontology file not found at {path}")
        raise
    except Exception as e:
        print(f"Error parsing ontology file {path}: {e}")
        traceback.print_exc()
        raise

def get_ancestors(g: Graph, term_uri: URIRef, prefix_map: Dict[str, str], visited_uris: set = None) -> List[str]:
    if visited_uris is None:
        visited_uris = set()

    ancestor_curies = set()
    # Find direct parents using rdfs:subClassOf
    # The object of rdfs:subClassOf is the parent class
    for parent_uri in g.objects(subject=term_uri, predicate=RDFS.subClassOf):
        if isinstance(parent_uri, URIRef) and parent_uri != OWL.Thing: # Ensure it's a URI and not owl:Thing
            if parent_uri not in visited_uris: # Avoid cycles and redundant processing
                visited_uris.add(parent_uri)
                parent_curie = uri_to_curie(parent_uri, prefix_map)
                if parent_curie and parent_curie != str(parent_uri): # Successfully converted to CURIE
                    ancestor_curies.add(parent_curie)
                    # Recursively get ancestors of this parent
                    ancestor_curies.update(get_ancestors(g, parent_uri, prefix_map, visited_uris))
    return list(ancestor_curies)


def extract_labels_and_synonyms(g: Graph) -> Dict[str, Dict[str, Any]]:
    data = {}
    # Consider all subjects that have a label or any OBOInOwl synonym property
    relevant_predicates = [
        RDFS.label,
        OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
        OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym
    ]
    
    processed_subjects = set() # To avoid processing a subject multiple times if it has many relevant predicates

    for pred in relevant_predicates:
        for s_uri in g.subjects(predicate=pred):
            if not isinstance(s_uri, URIRef) or s_uri in processed_subjects:
                continue # Skip literals, blank nodes, or already processed subjects
            
            curie = uri_to_curie(s_uri, NAMESPACE_MAP)
            if not curie or curie == str(s_uri): # Skip if not converted or not a FOODON curie (optional strictness)
                 if "FOODON:" not in curie and "IAO:" not in curie and "RO:" not in curie and "BFO:" not in curie: # Example filter
                    # print(f"Skipping non-ontology CURIE for labels/syns: {curie} from {s_uri}")
                    continue

            if curie not in data:
                data[curie] = {"label": None, "synonyms": []}

            label_val = g.value(subject=s_uri, predicate=RDFS.label)
            if label_val and isinstance(label_val, rdflib.Literal):
                data[curie]["label"] = str(label_val)

            synonyms = []
            for syn_prop in [OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
                             OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym]:
                for syn_obj in g.objects(subject=s_uri, predicate=syn_prop):
                    if isinstance(syn_obj, rdflib.Literal):
                        synonyms.append(str(syn_obj))
            if synonyms:
                 data[curie]["synonyms"].extend(s for s in synonyms if s not in data[curie]["synonyms"]) # Avoid duplicates

            processed_subjects.add(s_uri)
            
    # Clean up entries that ended up with no label and no synonyms
    final_data = {k: v for k, v in data.items() if v.get("label") or v.get("synonyms")}
    print(f"Extracted labels and synonyms for {len(final_data)} terms.")
    return final_data


def extract_definitions(g: Graph) -> Dict[str, str]:
    definitions = {}
    definition_prop_uri = IAO['0000115'] # IAO:0000115 is 'definition'
    for s_uri in g.subjects(predicate=definition_prop_uri):
        if not isinstance(s_uri, URIRef):
            continue
        
        curie = uri_to_curie(s_uri, NAMESPACE_MAP)
        if not curie or curie == str(s_uri) :
            if "FOODON:" not in curie and "IAO:" not in curie and "RO:" not in curie and "BFO:" not in curie:
                # print(f"Skipping non-ontology CURIE for definitions: {curie} from {s_uri}")
                continue
        
        def_obj = g.value(subject=s_uri, predicate=definition_prop_uri)
        if def_obj and isinstance(def_obj, rdflib.Literal):
            definitions[curie] = str(def_obj)
            
    print(f"Extracted definitions for {len(definitions)} terms.")
    return definitions

def extract_hierarchy(g: Graph) -> Dict[str, Dict[str, List[str]]]:
    hierarchy_data = {}
    # Consider all terms that are subjects or objects of rdfs:subClassOf triples
    # This ensures we capture all terms involved in the class hierarchy.
    all_terms_in_hierarchy = set()
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef): all_terms_in_hierarchy.add(s)
        if isinstance(o, URIRef): all_terms_in_hierarchy.add(o)
    
    for term_uri in all_terms_in_hierarchy:
        if term_uri == OWL.Thing: # Skip owl:Thing itself
            continue

        curie = uri_to_curie(term_uri, NAMESPACE_MAP)
        if not curie or curie == str(term_uri):
            if "FOODON:" not in curie and "IAO:" not in curie and "RO:" not in curie and "BFO:" not in curie:
                # print(f"Skipping non-ontology CURIE for hierarchy: {curie} from {term_uri}")
                continue

        # Direct Parents
        direct_parent_curies = []
        for parent_uri in g.objects(subject=term_uri, predicate=RDFS.subClassOf):
            if isinstance(parent_uri, URIRef) and parent_uri != OWL.Thing:
                parent_curie = uri_to_curie(parent_uri, NAMESPACE_MAP)
                if parent_curie and parent_curie != str(parent_uri): # Successfully converted
                    direct_parent_curies.append(parent_curie)
        
        # All Ancestors (transitive closure of parents)
        # Pass a new visited set for each term to avoid issues across calls
        ancestor_curies = get_ancestors(g, term_uri, NAMESPACE_MAP, visited_uris=set())
        
        if direct_parent_curies or ancestor_curies:
            hierarchy_data[curie] = {
                "parents": list(set(direct_parent_curies)), # Use set for uniqueness
                "ancestors": list(set(ancestor_curies))    # Use set for uniqueness
            }
            
    print(f"Extracted hierarchy data for {len(hierarchy_data)} terms.")
    return hierarchy_data


def extract_relations(g: Graph, props_to_extract: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    relations_data = {}
    
    for term_uri in g.subjects(unique=True): # Iterate over all unique subjects in the graph
        if not isinstance(term_uri, URIRef):
            continue

        curie = uri_to_curie(term_uri, NAMESPACE_MAP)
        if not curie or curie == str(term_uri):
             if "FOODON:" not in curie and "IAO:" not in curie and "RO:" not in curie and "BFO:" not in curie:
                # print(f"Skipping non-ontology CURIE for relations: {curie} from {term_uri}")
                continue

        term_specific_relations = {}
        for rel_name, rel_uri_str in props_to_extract.items():
            rel_uri = URIRef(rel_uri_str)
            target_curies = []
            for target_obj in g.objects(subject=term_uri, predicate=rel_uri):
                if isinstance(target_obj, URIRef): # Ensure target is a URI
                    target_curie = uri_to_curie(target_obj, NAMESPACE_MAP)
                    if target_curie and target_curie != str(target_obj): # Successfully converted
                        target_curies.append(target_curie)
            
            if target_curies:
                term_specific_relations[rel_name] = list(set(target_curies)) # Unique targets

        if term_specific_relations:
            relations_data[curie] = term_specific_relations
            
    print(f"Extracted relations for {len(relations_data)} terms based on specified properties.")
    return relations_data


def main():
    print("Starting ontology parsing...")
    data_dir = os.path.dirname(ONTOLOGY_DUMP_PATH)
    os.makedirs(data_dir, exist_ok=True)

    try:
        g = load_ontology(ONTOLOGY_FILE)

        print("\nExtracting data...")
        labels_synonyms = extract_labels_and_synonyms(g)
        definitions = extract_definitions(g)
        hierarchy = extract_hierarchy(g)
        relations = extract_relations(g, RELATION_PROPERTIES)

        print("\nMerging extracted data...")
        merged_data = {}
        all_curies = set(labels_synonyms.keys()) | \
                     set(definitions.keys()) | \
                     set(hierarchy.keys()) | \
                     set(relations.keys())

        for curie in all_curies:
            # Skip if the CURIE doesn't seem to belong to a relevant ontology (optional filter)
            # if not any(curie.startswith(p) for p in ["FOODON:", "IAO:", "RO:", "BFO:"]):
            #     # print(f"Skipping merge for non-primary CURIE: {curie}")
            #     continue

            merged_data[curie] = {
                "label": labels_synonyms.get(curie, {}).get("label"),
                "synonyms": labels_synonyms.get(curie, {}).get("synonyms", []),
                "definition": definitions.get(curie),
                "parents": hierarchy.get(curie, {}).get("parents", []),
                "ancestors": hierarchy.get(curie, {}).get("ancestors", []),
                "relations": relations.get(curie, {})
            }
        
        # Remove entries that are completely empty after merging
        final_merged_data = {}
        for curie, data_dict in merged_data.items():
            if data_dict.get("label") or \
               data_dict.get("synonyms") or \
               data_dict.get("definition") or \
               data_dict.get("parents") or \
               data_dict.get("ancestors") or \
               data_dict.get("relations"):
                final_merged_data[curie] = data_dict


        print(f"\nTotal merged entities with some data: {len(final_merged_data)}")

        print(f"Writing merged data to {ONTOLOGY_DUMP_PATH}")
        with open(ONTOLOGY_DUMP_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_merged_data, f, indent=4, ensure_ascii=False)

        print("Ontology parsing and data dump complete.")

    except FileNotFoundError:
        print(f"Parsing aborted: Ontology file not found at {ONTOLOGY_FILE}")
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()