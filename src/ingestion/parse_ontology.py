# src/ingestion/parse_ontology.py
"""
Parse an OWL ontology file into a JSON dump suitable for indexing.

Output format:
{
    "CURIE:12345": {
        "label": "term name",
        "synonyms": ["syn1", "syn2"],
        "definition": "A description of...",
        "parents": ["CURIE:00001"],
        "relations": [{"predicate": "has_part", "object": "CURIE:67890"}],
        "relations_text": "has_part: other term"
    }
}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL

from src import config
from src.utils.ontology_utils import uri_to_curie

logger = logging.getLogger(__name__)

# Standard annotation properties for extracting term metadata
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
OBO_IN_OWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")

# Common synonym properties
SYNONYM_PROPERTIES = [
    OBO_IN_OWL.hasExactSynonym,
    OBO_IN_OWL.hasRelatedSynonym,
    OBO_IN_OWL.hasBroadSynonym,
    OBO_IN_OWL.hasNarrowSynonym,
    SKOS.altLabel,
    SKOS.prefLabel,
]

# Common definition properties
DEFINITION_PROPERTIES = [
    IAO["0000115"],  # IAO:definition
    SKOS.definition,
    OBO_IN_OWL.hasDefinition,
    RDFS.comment,
]


def parse_ontology(
    ontology_path: Path,
    output_path: Path,
    curie_prefix_map: Optional[Dict[str, str]] = None,
    target_relations: Optional[List[str]] = None,
    relation_config: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Parse an OWL ontology file into a JSON dump.
    
    Parameters
    ----------
    ontology_path : Path
        Path to the OWL/OBO ontology file.
    output_path : Path
        Path where the JSON dump will be saved.
    curie_prefix_map : Optional[Dict[str, str]]
        Mapping of URI prefixes to CURIE prefixes.
        Default: config.CURIE_PREFIX_MAP
    target_relations : Optional[List[str]]
        List of relation CURIEs to extract (e.g., ["obo:BFO_0000050"]).
        Default: config.TARGET_RELATIONS_CURIES
    relation_config : Optional[Dict[str, Dict]]
        Mapping of relation CURIEs to labels.
        Default: config.RELATION_CONFIG
        
    Returns
    -------
    Dict[str, Any]
        The parsed ontology data (also saved to output_path).
    """
    # Apply defaults from config
    curie_prefix_map = curie_prefix_map or config.CURIE_PREFIX_MAP
    target_relations = target_relations or config.TARGET_RELATIONS_CURIES
    relation_config = relation_config or config.RELATION_CONFIG
    
    logger.info(f"Parsing ontology from: {ontology_path}")
    
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
    
    # Load the ontology
    g = Graph()
    logger.info("Loading ontology graph (this may take a while for large ontologies)...")
    g.parse(str(ontology_path))
    logger.info(f"Loaded {len(g)} triples")
    
    # Build a set of target relation URIs for fast lookup
    target_relation_uris: Set[URIRef] = set()
    for curie in target_relations:
        uri = _curie_to_uri_simple(curie, curie_prefix_map)
        if uri:
            target_relation_uris.add(uri)
    
    # Find all OWL classes
    ontology_data: Dict[str, Any] = {}
    classes = set(g.subjects(RDF.type, OWL.Class))
    logger.info(f"Found {len(classes)} OWL classes")
    
    for class_uri in classes:
        if not isinstance(class_uri, URIRef):
            continue
            
        curie = uri_to_curie(class_uri, curie_prefix_map)
        if curie == str(class_uri):
            # Could not convert to CURIE, skip
            continue
        
        # Extract label
        label = _get_literal(g, class_uri, RDFS.label)
        if not label:
            # Skip terms without labels
            continue
        
        # Extract synonyms
        synonyms: List[str] = []
        for prop in SYNONYM_PROPERTIES:
            for obj in g.objects(class_uri, prop):
                if isinstance(obj, Literal):
                    syn = str(obj).strip()
                    if syn and syn != label and syn not in synonyms:
                        synonyms.append(syn)
        
        # Extract definition
        definition = ""
        for prop in DEFINITION_PROPERTIES:
            definition = _get_literal(g, class_uri, prop)
            if definition:
                break
        
        # Extract parents (rdfs:subClassOf)
        parents: List[str] = []
        for parent_uri in g.objects(class_uri, RDFS.subClassOf):
            if isinstance(parent_uri, URIRef):
                parent_curie = uri_to_curie(parent_uri, curie_prefix_map)
                if parent_curie != str(parent_uri):
                    parents.append(parent_curie)
        
        # Extract relations
        relations: List[Dict[str, str]] = []
        relations_text_parts: List[str] = []
        
        for pred, obj in g.predicate_objects(class_uri):
            if pred in target_relation_uris and isinstance(obj, URIRef):
                pred_curie = uri_to_curie(pred, curie_prefix_map)
                obj_curie = uri_to_curie(obj, curie_prefix_map)
                
                # Get human-readable predicate label
                pred_label = pred_curie
                if pred_curie in relation_config:
                    pred_label = relation_config[pred_curie].get("label", pred_curie)
                
                # Get object label
                obj_label = _get_literal(g, obj, RDFS.label) or obj_curie
                
                relations.append({
                    "predicate": pred_curie,
                    "predicate_label": pred_label,
                    "object": obj_curie,
                    "object_label": obj_label,
                })
                relations_text_parts.append(f"{pred_label}: {obj_label}")
        
        ontology_data[curie] = {
            "label": label,
            "synonyms": synonyms,
            "definition": definition,
            "parents": parents,
            "relations": relations,
            "relations_text": "; ".join(relations_text_parts) if relations_text_parts else "",
        }
    
    logger.info(f"Extracted {len(ontology_data)} terms with labels")
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ontology_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved ontology dump to: {output_path}")
    return ontology_data


def _get_literal(g: Graph, subject: URIRef, predicate: URIRef) -> str:
    """Get the first literal value for a subject-predicate pair."""
    for obj in g.objects(subject, predicate):
        if isinstance(obj, Literal):
            return str(obj).strip()
    return ""


def _curie_to_uri_simple(curie: str, prefix_map: Dict[str, str]) -> Optional[URIRef]:
    """
    Simple CURIE to URI conversion.
    prefix_map is {uri_base: prefix}, so we need to invert it.
    """
    if ":" not in curie:
        return None
    
    prefix, local = curie.split(":", 1)
    
    # Find the base URI for this prefix
    for uri_base, pfx in prefix_map.items():
        if pfx == prefix:
            return URIRef(uri_base + local)
    
    return None


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.parse_ontology <ontology.owl> <output.json>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    parse_ontology(Path(sys.argv[1]), Path(sys.argv[2]))
