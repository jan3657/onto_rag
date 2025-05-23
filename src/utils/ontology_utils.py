# src/utils/ontology_utils.py
import rdflib
from rdflib import URIRef, Namespace
from typing import Optional, Dict, Union

# This import assumes that the script/module *importing* ontology_utils
# has already ensured that the project root (e.g., 'onto_rag') is on sys.path,
# so that 'src' is resolvable as a top-level package.
from src.config import NAMESPACE_MAP

def uri_to_curie(uri: Union[str, URIRef], namespace_map: Dict[str, str] = NAMESPACE_MAP) -> str:
    """Converts a full URI to a CURIE (e.g., http://...#term -> PREFIX:term)."""
    uri_str = str(uri) # Ensure it's a string
    for prefix, base_uri in namespace_map.items():
        if uri_str.startswith(base_uri):
            return f"{prefix}:{uri_str[len(base_uri):]}"

    # Fallback for common RDF/RDFS/OWL/XSD prefixes if not in map (or if map doesn't have them as strings)
    common_rdf_prefixes = {
        "rdf": str(rdflib.RDF),
        "rdfs": str(rdflib.RDFS),
        "owl": str(rdflib.OWL),
        "xsd": str(rdflib.XSD),
    }
    for prefix, base_uri in common_rdf_prefixes.items():
        if uri_str.startswith(base_uri):
            return f"{prefix}:{uri_str[len(base_uri):]}"
            
    # If rdflib can make a qname (usually for registered namespaces)
    try:
        g = rdflib.Graph()
        # Bind known namespaces to help compute_qname
        for prefix, ns_uri_str in namespace_map.items():
            g.bind(prefix, Namespace(ns_uri_str))
        g.bind("owl", rdflib.OWL)
        g.bind("rdf", rdflib.RDF)
        g.bind("rdfs", rdflib.RDFS)
        g.bind("xsd", rdflib.XSD)

        qname = g.compute_qname(URIRef(uri_str)) # Returns (prefix, namespace, name)
        return f"{qname[0]}:{qname[2]}"
    except: # noqa
        pass # If rdflib fails, just return the original URI string

    return uri_str # If no CURIE conversion possible, return original URI string

def curie_to_uri(curie: str, namespace_map: Dict[str, str] = NAMESPACE_MAP) -> Optional[URIRef]:
    """Converts a CURIE (e.g., PREFIX:term) to a full rdflib.URIRef."""
    if ':' not in curie:
        return None # Not a valid CURIE format

    prefix, local_name = curie.split(':', 1)
    base_uri = namespace_map.get(prefix)

    if base_uri:
        return URIRef(base_uri + local_name)
    else:
        # Try common RDF prefixes if not in custom map
        common_rdf_prefixes_to_ns = {
            "rdf": rdflib.RDF,
            "rdfs": rdflib.RDFS,
            "owl": rdflib.OWL,
            "xsd": rdflib.XSD,
        }
        if prefix in common_rdf_prefixes_to_ns:
            return URIRef(common_rdf_prefixes_to_ns[prefix][local_name])
        return None # Prefix not found