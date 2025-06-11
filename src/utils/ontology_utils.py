# src/utils/ontology_utils.py
import rdflib
from rdflib import URIRef, Namespace
from typing import Optional, Dict, Union

# Adjusted: Import CURIE_PREFIX_MAP and use it as the default.
# This assumes that the script/module *importing* ontology_utils
# has already ensured that the project root (e.g., 'onto_rag') is on sys.path,
# so that 'src' is resolvable as a top-level package.
from src.config import CURIE_PREFIX_MAP # Corrected import

def uri_to_curie(uri: Union[str, URIRef], namespace_map: Dict[str, str] = CURIE_PREFIX_MAP) -> str:
    """
    Converts a full URI to a CURIE (e.g., http://...#term -> PREFIX:term).
    Assumes namespace_map is structured as {base_uri_str: prefix_str}.
    """
    uri_str = str(uri) # Ensure it's a string

    # Iterate through the provided namespace_map (base_uri: prefix)
    # Sort by length of base_uri descending to match longest first (more specific)
    # This helps avoid issues where one base_uri is a prefix of another.
    # e.g., "http://purl.obolibrary.org/obo/" and "http://purl.obolibrary.org/obo/FOODON_"
    sorted_namespace_map_items = sorted(namespace_map.items(), key=lambda item: len(item[0]), reverse=True)

    for base_uri, prefix in sorted_namespace_map_items:
        if uri_str.startswith(base_uri):
            return f"{prefix}:{uri_str[len(base_uri):]}"

    # Fallback for common RDF/RDFS/OWL/XSD prefixes if not found in the main map
    # This local map is prefix: base_uri_str
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
        # For namespace_map (base_uri: prefix), we need to iterate as base_uri, prefix
        for ns_uri_str_from_map, prefix_from_map in namespace_map.items():
             g.bind(prefix_from_map, Namespace(ns_uri_str_from_map))

        # Bind common RDF ones too, in case they weren't in namespace_map or to ensure standard prefixes
        g.bind("owl", rdflib.OWL)
        g.bind("rdf", rdflib.RDF)
        g.bind("rdfs", rdflib.RDFS)
        g.bind("xsd", rdflib.XSD)

        # compute_qname might fail if the URI doesn't match any bound namespace's base
        # It returns (prefix, namespace_uri, local_name)
        qname_tuple = g.compute_qname(URIRef(uri_str))
        return f"{qname_tuple[0]}:{qname_tuple[2]}"
    except Exception: # Broad except as compute_qname can raise various things or return unexpected tuples
        pass # If rdflib fails, just return the original URI string

    return uri_str # If no CURIE conversion possible, return original URI string

def curie_to_uri(curie: str, namespace_map: Dict[str, str] = CURIE_PREFIX_MAP) -> Optional[URIRef]:
    """
    Converts a CURIE (e.g., PREFIX:term) to a full rdflib.URIRef.
    Assumes namespace_map is structured as {base_uri_str: prefix_str}.
    """
    if ':' not in curie:
        # Try to see if it's a default rdflib qname like "rdf:type" that rdflib can expand
        # This part might be less common if CURIEs are always expected with user-defined prefixes
        try:
            g = rdflib.Graph()
            # Bind namespaces from the map (base_uri: prefix)
            for ns_uri_str_from_map, prefix_from_map in namespace_map.items():
                g.bind(prefix_from_map, Namespace(ns_uri_str_from_map))
            # Bind common RDF ones
            g.bind("owl", rdflib.OWL)
            g.bind("rdf", rdflib.RDF)
            g.bind("rdfs", rdflib.RDFS)
            g.bind("xsd", rdflib.XSD)
            
            # If it's something like "owl:Class", g.namespace_manager.expand_curie will work
            expanded_uri = g.namespace_manager.expand_curie(curie)
            if str(expanded_uri) != curie: # Check if expansion actually happened
                 return URIRef(expanded_uri)
        except Exception:
            pass # If expansion fails, proceed to manual lookup
        return None # Not a valid CURIE format for our map, and rdflib couldn't expand

    prefix_part, local_name = curie.split(':', 1)

    # Iterate through namespace_map (base_uri: prefix) to find the matching prefix
    found_base_uri = None
    for base_uri_key, prefix_val in namespace_map.items():
        if prefix_val == prefix_part:
            found_base_uri = base_uri_key
            break
    
    if found_base_uri:
        return URIRef(found_base_uri + local_name)
    else:
        # Fallback: Try common RDF prefixes if not in the custom map
        # This local map is prefix: rdflib.Namespace object
        common_rdf_namespaces = {
            "rdf": rdflib.RDF,
            "rdfs": rdflib.RDFS,
            "owl": rdflib.OWL,
            "xsd": rdflib.XSD,
        }
        if prefix_part in common_rdf_namespaces:
            # Access items in the namespace like attributes: common_rdf_namespaces[prefix_part].type
            # or by string concatenation: URIRef(str(common_rdf_namespaces[prefix_part]) + local_name)
            return URIRef(str(common_rdf_namespaces[prefix_part]) + local_name)
        return None # Prefix not found in custom map or common RDF prefixes