import pytest
import os
import json
import tempfile
import shutil
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Adjust path to import modules from src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.parse_ontology import main as parse_ontology_main # Need to run parsing first
from ingestion.build_lexical_index import build_index
import config # Need config to override paths

# Define a known CURIE and search terms from the test snippet
TEST_CURIE = "FOODON:00001100" # apple
TEST_LABEL = "apple"
TEST_SYNONYM = "eating apple"
TEST_DEFINITION_KEYWORD = "fruit" # Keyword from definition "The fruit of a Malus domestica tree."
TEST_RELATION_KEYWORD = "apple tree" # Keyword from relations_text (target label)


# Fixture to create a temporary directory for the test index
@pytest.fixture(scope="module")
def temp_index_dir():
    """Creates a temporary directory for the Whoosh index."""
    path = tempfile.mkdtemp(prefix="whoosh_test_index_")
    print(f"Created temp index dir: {path}")
    yield path
    print(f"Cleaning up temp index dir: {path}")
    shutil.rmtree(path)

# Fixture to run the full ingestion pipeline on the test snippet and build index
@pytest.fixture(scope="module")
def built_test_index(temp_index_dir):
    """Runs parse_ontology and build_index on the test snippet."""
    # Use temp files/dirs for test data and index
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
        test_json_path = tmp_json.name

    # Temporarily override config paths for the test
    original_ontology_file = config.ONTOLOGY_FILE
    original_dump_path = config.ONTOLOGY_DUMP_PATH
    original_index_dir = config.WHOOSH_INDEX_DIR

    config.ONTOLOGY_FILE = config.TEST_ONTOLOGY_FILE # Use the small test snippet
    config.ONTOLOGY_DUMP_PATH = test_json_path
    config.WHOOSH_INDEX_DIR = temp_index_dir

    try:
        # 1. Run parsing
        print("\nRunning parse_ontology_main for test index build...")
        parse_ontology_main()
        assert os.path.exists(test_json_path), "Ontology dump JSON not created!"

        # 2. Run index building
        print("\nRunning build_index for test index build...")
        build_index(test_json_path, temp_index_dir)
        assert os.path.exists(os.path.join(temp_index_dir, 'SCHEMA')), "Whoosh index not created!"

        # Yield the index directory path
        yield temp_index_dir

    finally:
        # Restore original paths
        config.ONTOLOGY_FILE = original_ontology_file
        config.ONTOLOGY_DUMP_PATH = original_dump_path
        config.WHOOSH_INDEX_DIR = original_index_dir
        # Clean up temporary JSON file
        if os.path.exists(test_json_path):
            os.unlink(test_json_path)


def test_index_exists(built_test_index):
    """Tests if the index directory was created and contains index files."""
    assert os.path.exists(built_test_index)
    assert os.path.exists(os.path.join(built_test_index, 'SCHEMA'))
    assert os.path.exists(os.path.join(built_test_index, 'MAIN')) # Or other index files

def test_query_by_label_exact(built_test_index):
    """Tests exact search on the label field."""
    ix = open_dir(built_test_index)
    searcher = ix.searcher()
    # Query the 'label' field
    parser = QueryParser("label", ix.schema)
    query = parser.parse(TEST_LABEL) # "apple"
    results = searcher.search(query)

    print(f"\nSearch results for '{TEST_LABEL}': {results}")

    assert len(results) > 0, f"No results found for '{TEST_LABEL}'"
    # Check if the expected CURIE is among the results, preferably the top one
    assert results[0]['curie'] == TEST_CURIE
    assert results[0]['label'] == TEST_LABEL # Verify stored label

    searcher.close()
    ix.close() # Close index readers/writers

def test_query_by_label_fuzzy(built_test_index):
    """Tests fuzzy search on the label field."""
    ix = open_dir(built_test_index)
    searcher = ix.searcher()
    parser = QueryParser("label", ix.schema)
    # Fuzzy search for "appl" (e.g. "appl~")
    query = parser.parse("appl~") # Whoosh fuzzy syntax
    results = searcher.search(query)

    print(f"\nFuzzy search results for 'appl~': {results}")

    assert len(results) > 0, "No results found for 'appl~'"
    assert results[0]['curie'] == TEST_CURIE
    assert results[0]['label'] == TEST_LABEL

    searcher.close()
    ix.close()

def test_query_by_synonym(built_test_index):
    """Tests search on the synonyms field."""
    ix = open_dir(built_test_index)
    searcher = ix.searcher()
    # Query the 'synonyms' field
    parser = QueryParser("synonyms", ix.schema)
    query = parser.parse(TEST_SYNONYM) # "eating apple"
    results = searcher.search(query)

    print(f"\nSearch results for synonym '{TEST_SYNONYM}': {results}")

    assert len(results) > 0, f"No results found for synonym '{TEST_SYNONYM}'"
    assert results[0]['curie'] == TEST_CURIE

    searcher.close()
    ix.close()

def test_query_by_definition_keyword(built_test_index):
    """Tests search on the definition field."""
    ix = open_dir(built_test_index)
    searcher = ix.searcher()
    # Query the 'definition' field
    parser = QueryParser("definition", ix.schema)
    query = parser.parse(TEST_DEFINITION_KEYWORD) # "fruit"
    results = searcher.search(query)

    print(f"\nSearch results for definition keyword '{TEST_DEFINITION_KEYWORD}': {results}")

    assert len(results) > 0, f"No results found for definition keyword '{TEST_DEFINITION_KEYWORD}'"
    assert results[0]['curie'] == TEST_CURIE

    searcher.close()
    ix.close()

def test_query_by_relations_keyword(built_test_index):
    """Tests search on the flattened relations_text field."""
    ix = open_dir(built_test_index)
    searcher = ix.searcher()
    # Query the 'relations_text' field
    parser = QueryParser("relations_text", ix.schema)
    # Search for a keyword expected in the relations_text (e.g., target's label or part of target's CURIE)
    # Note: Indexing targets as just CURIEs requires searching for 'FOODON:00001101'
    # Indexing target labels requires adding that logic in build_lexical_index
    # Let's assume for now we search for the CURIE or part of it if indexed as text
    # Or if we enhance build_index to add target labels to relations_text:
    # relations_text = " ".join([f"{prop}: {' '.join(targets)} {' '.join(target_labels)}" for ...])
    # For simplicity with current build_index, let's search the target CURIE part
    query = parser.parse("00001101") # Search part of the target CURIE FOODON:00001101
    results = searcher.search(query)

    print(f"\nSearch results for relations keyword '00001101': {results}")

    assert len(results) > 0, f"No results found for relations keyword '00001101'"
    assert results[0]['curie'] == TEST_CURIE

    searcher.close()
    ix.close()