import pytest
import os
import json
import rdflib
from rdflib import Graph

from src.ingestion.parse_ontology import (
    load_ontology,
    extract_labels_and_synonyms,
    extract_definitions,
    extract_hierarchy,
    extract_relations,
)
from src.config import TEST_ONTOLOGY_FILE, NAMESPACE_MAP, RELATION_PROPERTIES
from src.utils.ontology_utils import uri_to_curie  # Import if needed for assertions

# Define a fixture to load the test graph once for all tests
@pytest.fixture(scope="module")
def test_graph():
    """Loads the test ontology snippet into a graph."""
    if not os.path.exists(TEST_ONTOLOGY_FILE):
        pytest.skip(f"Test ontology snippet not found at {TEST_ONTOLOGY_FILE}")
    try:
        return load_ontology(TEST_ONTOLOGY_FILE)
    except Exception as e:
        pytest.fail(f"Failed to load test ontology: {e}")

# Define a known CURIE from the test snippet
TEST_CURIE = "FOODON:00001100" # apple
TEST_CURIE_PARENT1 = "FOODON:00001000" # plant-based food
TEST_CURIE_PARENT2 = "FOODON:00002000" # Pome fruit
TEST_CURIE_RELATION_TARGET = "FOODON:00001101" # apple tree (example target)
TEST_RELATION_NAME = "part_of" # example relation name

def test_load_ontology(test_graph):
    """Tests if the ontology loads and is an RDFLib Graph."""
    assert isinstance(test_graph, Graph)
    assert len(test_graph) > 0, "Test graph is empty!"
    print(f"Loaded test graph with {len(test_graph)} triples.")


def test_extract_labels_and_synonyms(test_graph):
    """Tests extraction of labels and synonyms."""
    labels_synonyms = extract_labels_and_synonyms(test_graph)
    print(f"Extracted labels/synonyms: {labels_synonyms}")

    assert TEST_CURIE in labels_synonyms
    apple_data = labels_synonyms[TEST_CURIE]
    assert apple_data['label'] == "apple"
    assert "eating apple" in apple_data['synonyms']
    assert "apple fruit" in apple_data['synonyms']
    assert TEST_CURIE_PARENT1 in labels_synonyms # Should also extract its label/synonyms
    assert labels_synonyms[TEST_CURIE_PARENT1]['label'] == "plant-based food"


def test_extract_definitions(test_graph):
    """Tests extraction of definitions."""
    definitions = extract_definitions(test_graph)
    print(f"Extracted definitions: {definitions}")

    assert TEST_CURIE in definitions
    assert "The fruit of a Malus domestica tree." in definitions[TEST_CURIE] # Use 'in' for substring check or exact match


def test_extract_hierarchy(test_graph):
    """Tests extraction of hierarchy (parents and ancestors)."""
    hierarchy = extract_hierarchy(test_graph)
    print(f"Extracted hierarchy: {hierarchy}")

    assert TEST_CURIE in hierarchy
    apple_hierarchy = hierarchy[TEST_CURIE]

    # Check direct parents
    assert TEST_CURIE_PARENT1 in apple_hierarchy['parents']
    assert TEST_CURIE_PARENT2 in apple_hierarchy['parents']
    assert len(apple_hierarchy['parents']) >= 2 # At least 2 parents from the snippet

    # Check ancestors (should include parents and parents' ancestors)
    assert TEST_CURIE_PARENT1 in apple_hierarchy['ancestors']
    assert TEST_CURIE_PARENT2 in apple_hierarchy['ancestors']
    # Assuming plant-based food has no ancestors in this snippet, ancestors == parents here
    # If plant-based food had parents, they should appear in apple's ancestors.
    # For this snippet, ancestors should be the same as parents.
    assert set(apple_hierarchy['ancestors']) == set(apple_hierarchy['parents'])


def test_extract_relations(test_graph):
    """Tests extraction of specific relations."""
    relations = extract_relations(test_graph, RELATION_PROPERTIES)
    print(f"Extracted relations: {relations}")

    assert TEST_CURIE in relations
    apple_relations = relations[TEST_CURIE]

    # Check if the specific relation from the snippet was found
    assert TEST_RELATION_NAME in apple_relations
    assert TEST_CURIE_RELATION_TARGET in apple_relations[TEST_RELATION_NAME]
    assert len(apple_relations[TEST_RELATION_NAME]) >= 1


# You could add a test that runs the full main parsing flow using the test snippet
# and checks the structure/content of the resulting JSON file.
# Example (requires a temporary file):
# import tempfile
# @pytest.fixture
# def temp_json_path():
#     with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
#         yield tmp.name
#     os.unlink(tmp.name)
#
# def test_main_parsing_flow(temp_json_path):
#      # Temporarily override config paths for the test
#      original_ontology_file = config.ONTOLOGY_FILE
#      original_dump_path = config.ONTOLOGY_DUMP_PATH
#      config.ONTOLOGY_FILE = TEST_ONTOLOGY_FILE
#      config.ONTOLOGY_DUMP_PATH = temp_json_path
#
#      try:
#          parse_ontology.main()
#          assert os.path.exists(temp_json_path)
#          with open(temp_json_path, 'r', encoding='utf-8') as f:
#              data = json.load(f)
#
#          assert TEST_CURIE in data
#          apple_data = data[TEST_CURIE]
#          assert apple_data['label'] == 'apple'
#          assert 'synonyms' in apple_data and len(apple_data['synonyms']) > 0
#          assert 'definition' in apple_data and apple_data['definition'] is not None
#          assert 'parents' in apple_data and len(apple_data['parents']) > 0
#          assert 'ancestors' in apple_data and len(apple_data['ancestors']) > 0
#          assert 'relations' in apple_data and len(apple_data['relations']) > 0
#
#      finally:
#          # Restore original paths
#          config.ONTOLOGY_FILE = original_ontology_file
#          config.ONTOLOGY_DUMP_PATH = original_dump_path