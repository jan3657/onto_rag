# scripts/find_namespaces.py
import rdflib
from rdflib import URIRef
from collections import Counter
import re
import os
import sys

# Add project root to allow importing src.config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.config import FOODON_PATH # Or any other OWL file you want to inspect

def get_base_uri(uri_str):
    """
    Tries to extract a base URI from a full URI.
    e.g., http://purl.obolibrary.org/obo/FOODON_0000123 -> http://purl.obolibrary.org/obo/FOODON_
    e.g., http://www.w3.org/2000/01/rdf-schema#label -> http://www.w3.org/2000/01/rdf-schema#
    """
    if '#' in uri_str:
        return uri_str.rsplit('#', 1)[0] + '#'
    # Common OBO pattern: ends with an underscore followed by digits
    match_obo = re.match(r'(.+[_A-Z]+_)(\d+)$', uri_str)
    if match_obo:
        return match_obo.group(1)
    # General pattern: up to the last slash
    if '/' in uri_str:
        return uri_str.rsplit('/', 1)[0] + '/'
    return None

def main(ontology_path):
    print(f"Loading ontology from: {ontology_path}...")
    g = rdflib.Graph()
    try:
        g.parse(ontology_path) # format will be auto-detected
        print(f"Successfully parsed. Found {len(g)} triples.")
    except Exception as e:
        print(f"Error parsing ontology: {e}")
        return

    uris = set()
    for s, p, o in g:
        if isinstance(s, URIRef):
            uris.add(str(s))
        if isinstance(p, URIRef):
            uris.add(str(p))
        if isinstance(o, URIRef):
            uris.add(str(o))

    print(f"\nFound {len(uris)} unique URIs.")

    base_uri_counts = Counter()
    for uri in uris:
        base = get_base_uri(uri)
        if base:
            base_uri_counts[base] += 1

    print("\nPotential Base URIs (with counts of distinct full URIs using them):")
    # Sort by count descending
    for base, count in base_uri_counts.most_common():
        print(f"- \"{base}\": (used in {count} URIs)")

    print("\n--- rdflib's known namespaces (Prefix: Namespace) ---")
    for prefix, namespace in g.namespaces():
        print(f"  {prefix}: {namespace}")


if __name__ == "__main__":
    # You can make the ontology path an argument if you like
    # For now, using FOODON_PATH from config
    if not os.path.exists(FOODON_PATH):
        print(f"ERROR: FoodON ontology file not found at {FOODON_PATH}")
    else:
        main(FOODON_PATH)