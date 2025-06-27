# scripts/find_namespaces.py
import rdflib
from rdflib import URIRef
from collections import Counter
import re
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ONTOLOGIES_CONFIG

def get_base_uri(uri_str: str) -> Optional[str]:
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

def analyze_ontology(ontology_path: Path):
    """Loads a single ontology and prints a report of its namespaces."""
    print(f"\nLoading ontology from: {ontology_path}...")
    g = rdflib.Graph()
    try:
        # rdflib's parse method can handle Path objects directly
        g.parse(ontology_path) # format will be auto-detected
        print(f"Successfully parsed. Found {len(g)} triples.")
    except Exception as e:
        print(f"ERROR: Could not parse ontology: {e}")
        return

    uris = set()
    for s, p, o in g:
        if isinstance(s, URIRef): uris.add(str(s))
        if isinstance(p, URIRef): uris.add(str(p))
        if isinstance(o, URIRef): uris.add(str(o))

    print(f"Found {len(uris)} unique URIs.")

    base_uri_counts = Counter()
    for uri in uris:
        base = get_base_uri(uri)
        if base:
            base_uri_counts[base] += 1

    print("\n--- Potential Base URIs (with usage counts) ---")
    if not base_uri_counts:
        print("No common base URIs found.")
    else:
        # Sort by count descending and print
        for base, count in base_uri_counts.most_common():
            print(f"- \"{base}\": (used in {count} distinct URIs)")

    print("\n--- rdflib's known prefixes from file ---")
    namespaces = list(g.namespaces())
    if not namespaces:
        print("No prefixes were explicitly defined in the ontology file.")
    else:
        for prefix, namespace in namespaces:
            print(f"  {prefix}: {namespace}")

def main():
    """
    Loops through all ontologies defined in the project configuration
    and runs the namespace analysis on each one.
    """
    print("--- Finding Namespaces in All Configured Ontologies ---")

    if not ONTOLOGIES_CONFIG:
        print("No ontologies found in the ONTOLOGIES_CONFIG dictionary in your config file. Exiting.")
        return

    # Loop through each configured ontology
    for name, config_data in ONTOLOGIES_CONFIG.items():
        print(f"\n==================================================")
        print(f"Analyzing Ontology: '{name}'")
        print(f"==================================================")

        ontology_path = config_data.get('path')

        # Check if the path is defined in the config
        if not ontology_path:
            print(f"WARNING: No 'path' key found in configuration for '{name}'. Skipping.")
            continue

        # Check if the file actually exists using pathlib
        if not ontology_path.exists():
            print(f"ERROR: Ontology file for '{name}' not found at the configured path: '{ontology_path}'. Skipping.")
            continue

        # Call the analysis function for the valid path
        analyze_ontology(ontology_path)

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()