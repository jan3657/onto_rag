# src/ingestion/parse_ontology.py
import logging
import rdflib
from rdflib import Graph, Namespace, URIRef, RDFS, OWL, RDF
from typing import Dict, List, Any
import json
import traceback
from pathlib import Path
import re
from rdflib import SKOS

from src.utils.logging_config import setup_run_logging

# Now import using the 'src' package prefix
from src.config import (
    ONTOLOGIES_CONFIG,       
    CURIE_PREFIX_MAP,
    RELATION_CONFIG,
    TARGET_RELATIONS_CURIES,
    IAO_NS_STR,
    OBOINOWL_NS_STR,
)
from src.utils.ontology_utils import uri_to_curie, curie_to_uri

# --- Logging Setup ---
setup_run_logging()
logger = logging.getLogger(__name__)


# Define commonly used namespaces (can still use these locally for convenience)
IAO = Namespace(IAO_NS_STR)
OBOINOWL = Namespace(OBOINOWL_NS_STR)

def make_curie_validator(ontology_name: str):
    pat = ONTOLOGIES_CONFIG[ontology_name].get('id_pattern')
    if not pat:
        # fallback: accept '<PREFIX>:<alnum>' if not configured
        pref = ONTOLOGIES_CONFIG[ontology_name]['prefix']
        return re.compile(rf'^{re.escape(pref)}[A-Za-z0-9_]+$').match
    return re.compile(pat).match

def is_valid_curie_for_ontology(curie: str, ontology_name: str) -> bool:
    matcher = make_curie_validator(ontology_name)
    return bool(matcher(curie))

def get_valid_entities(g: Graph,
                       include_skos_concepts: bool = True) -> set[URIRef]:
    """
    Returns the set of URIs that are 'terms' you want to index.
    - Accepts owl:Class and (optionally) skos:Concept
    - Excludes properties, ontology headers, individuals by default
    """
    classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    skos_concepts = set()
    if include_skos_concepts:
        skos_concepts = {s for s in g.subjects(RDF.type, SKOS.Concept) if isinstance(s, URIRef)}

    # known non-term types to exclude
    excluded_types = {
        OWL.ObjectProperty, OWL.AnnotationProperty, OWL.DatatypeProperty,
        RDF.Property, OWL.Ontology, OWL.NamedIndividual  # add more if you bump into them
    }
    excluded_subjects = set()
    for t in excluded_types:
        excluded_subjects.update({s for s in g.subjects(RDF.type, t)})

    valid = (classes | skos_concepts) - excluded_subjects
    return valid


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


def extract_labels_and_synonyms(g: Graph, prefix_map: Dict[str, str], valid_entities: set) -> Dict[str, Dict[str, Any]]:
    data = {}
    relevant_predicates = [
        RDFS.label,
        OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
        OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym
    ]
    
    # Only process entities that are in the valid_entities gate
    for s_uri in valid_entities:
        curie = uri_to_curie(s_uri, prefix_map)
        if not curie or curie == str(s_uri):
            continue

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
        
        data[curie]["synonyms"] = current_synonyms
            
    final_data = {k: v for k, v in data.items() if v.get("label") or v.get("synonyms")}
    logger.info(f"Extracted labels and synonyms for {len(final_data)} terms.")
    return final_data


def extract_definitions(g: Graph, prefix_map: Dict[str, str], valid_entities: set) -> Dict[str, str]:
    definitions = {}
    definition_prop_uri = IAO['0000115'] # IAO:0000115 is 'definition'
    
    # Only process entities that are in the valid_entities gate
    for s_uri in valid_entities:
        curie = uri_to_curie(s_uri, prefix_map)
        if not curie or curie == str(s_uri):
            continue
        
        def_obj = g.value(subject=s_uri, predicate=definition_prop_uri)
        if def_obj and isinstance(def_obj, rdflib.Literal):
            definitions[curie] = str(def_obj)
            
    logger.info(f"Extracted definitions for {len(definitions)} terms.")
    return definitions


def extract_hierarchy(g: Graph, prefix_map: Dict[str, str], valid_entities: set) -> Dict[str, Dict[str, List[str]]]:
    hierarchy_data = {}
    
    # Only process entities that are in the valid_entities gate
    for term_uri in valid_entities:
        if term_uri == OWL.Thing:
            continue

        curie = uri_to_curie(term_uri, prefix_map)
        if not curie or curie == str(term_uri):
            continue

        direct_parent_curies = []
        for parent_uri in g.objects(subject=term_uri, predicate=RDFS.subClassOf):
            if isinstance(parent_uri, URIRef) and parent_uri != OWL.Thing:
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


def extract_relations(g: Graph, props_to_extract: Dict[str, str], prefix_map: Dict[str, str], valid_entities: set) -> Dict[str, Dict[str, List[str]]]:
    relations_data = {}
    
    # Only process entities that are in the valid_entities gate
    for term_uri in valid_entities:
        curie = uri_to_curie(term_uri, prefix_map)
        if not curie or curie == str(term_uri):
            continue

        term_specific_relations = {}
        for rel_readable_name, rel_uri_str in props_to_extract.items():
            rel_uri = URIRef(rel_uri_str)
            target_curies = []
            for target_obj in g.objects(subject=term_uri, predicate=rel_uri):
                if isinstance(target_obj, URIRef):
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
    logger.info("--- Starting Ontology Parsing for Each Configured Ontology ---")

    # Prepare relation properties once
    relation_properties_for_extraction = {}
    for rel_curie_str in TARGET_RELATIONS_CURIES:
        if rel_curie_str in RELATION_CONFIG:
            readable_name = RELATION_CONFIG[rel_curie_str].get("label", rel_curie_str)
            full_rel_uri = curie_to_uri(rel_curie_str, CURIE_PREFIX_MAP)
            if full_rel_uri:
                relation_properties_for_extraction[readable_name] = str(full_rel_uri)
            else:
                logger.warning(f"Could not convert relation CURIE {rel_curie_str} to URI. Skipping this relation.")
        else:
            logger.warning(f"Relation CURIE {rel_curie_str} from TARGET_RELATIONS_CURIES not found in RELATION_CONFIG.")

    # Loop through each ontology configured in config.py
    for name, config_data in ONTOLOGIES_CONFIG.items():
        ontology_path = config_data['path']
        dump_path = config_data['dump_json_path']
        
        # Ensure the output directory exists
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n--- Processing Ontology: '{name}' ---")
        logger.info(f"Source: {ontology_path}")
        logger.info(f"Destination: {dump_path}")

        if not ontology_path.exists():
            logger.error(f"Ontology file not found. Skipping '{name}'.")
            continue

        try:
            # 1. Load the single ontology graph
            g = load_ontology(ontology_path)

            # 2. Get the valid entities gate for this ontology
            valid_entities = get_valid_entities(g, include_skos_concepts=True)
            logger.info(f"Found {len(valid_entities)} valid entities to process in '{name}'.")

            # 3. Extract data FROM THIS GRAPH ONLY using the valid entities gate
            logger.info(f"Extracting data for '{name}'...")
            labels_synonyms = extract_labels_and_synonyms(g, CURIE_PREFIX_MAP, valid_entities)
            definitions = extract_definitions(g, CURIE_PREFIX_MAP, valid_entities)
            hierarchy = extract_hierarchy(g, CURIE_PREFIX_MAP, valid_entities)
            relations = extract_relations(g, relation_properties_for_extraction, CURIE_PREFIX_MAP, valid_entities)

            # 4. Merge extracted data for this ontology
            logger.info("Merging extracted data...")
            ontology_specific_data = {}
            all_curies = set(labels_synonyms.keys()) | set(definitions.keys()) | set(hierarchy.keys()) | set(relations.keys())

            for curie_key in all_curies:
                ontology_specific_data[curie_key] = {
                    "label": labels_synonyms.get(curie_key, {}).get("label"),
                    "synonyms": labels_synonyms.get(curie_key, {}).get("synonyms", []),
                    "definition": definitions.get(curie_key),
                    "parents": hierarchy.get(curie_key, {}).get("parents", []),
                    "ancestors": hierarchy.get(curie_key, {}).get("ancestors", []),
                    "relations": relations.get(curie_key, {})
                }
            
            ont_name = name  # loop variable
            final_data = {
                k: v for k, v in ontology_specific_data.items()
                if is_valid_curie_for_ontology(k, ont_name) and any(v.values())
            }
                    
            # 5. Save the dedicated dump file
            logger.info(f"Found {len(final_data)} entities with data in '{name}'.")
            logger.info(f"Writing data to {dump_path}")
            with open(dump_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Successfully processed '{name}'.")

        except Exception as e:
            logger.error(f"An error occurred while processing '{name}': {e}")
            traceback.print_exc()

    logger.info("\n--- All Ontology Parsing Complete ---")


if __name__ == "__main__":
    main()