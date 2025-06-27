# src/config.py
import os
from dotenv import load_dotenv
from rdflib import Namespace # This is fine, though rdflib.Namespace is not directly used for string constants below.

# Project Root Directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "."))
# print(f"Project root directory: {PROJECT_ROOT}") # Keep for debugging if you like

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Data Directory (for ontology dump, indexes, etc.)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Ontologies Directory
ONTOLOGIES_DIR = os.path.join(PROJECT_ROOT, "ontologies")
FOODON_PATH = os.path.join(ONTOLOGIES_DIR, "foodon-with-syns.owl")
CHEBI_PATH = os.path.join(ONTOLOGIES_DIR, "chebi.owl")
TEST_FOODON_SNIPPET_PATH = os.path.join(ONTOLOGIES_DIR, "test_foodon_snippet.owl")

# --- NEW: Central Ontologies Configuration ---
# The keys ('foodon', 'chebi') are used as identifiers throughout the pipeline.
ONTOLOGIES_CONFIG = {
    'foodon': {
        'path': os.path.join(ONTOLOGIES_DIR, "foodon.owl"),
        'prefix': 'FOODON:',
        'dump_json_path': os.path.join(DATA_DIR, "ontology_dump_foodon.json"),
        'enriched_docs_path': os.path.join(DATA_DIR, f"enriched_documents_foodon.json"),
        'embeddings_path': os.path.join(DATA_DIR, f"embeddings_foodon.json"),
        'whoosh_index_dir': os.path.join(DATA_DIR, f"whoosh_index_foodon"),
        'faiss_index_path': os.path.join(DATA_DIR, f"faiss_index_foodon.bin"),
        'faiss_metadata_path': os.path.join(DATA_DIR, f"faiss_metadata_foodon.json"),
    },
    'chebi': {
        'path': os.path.join(ONTOLOGIES_DIR, "chebi.owl"),
        'prefix': 'CHEBI:',
        'dump_json_path': os.path.join(DATA_DIR, "ontology_dump_chebi.json"),
        'enriched_docs_path': os.path.join(DATA_DIR, f"enriched_documents_chebi.json"),
        'embeddings_path': os.path.join(DATA_DIR, f"embeddings_chebi.json"),
        'whoosh_index_dir': os.path.join(DATA_DIR, f"whoosh_index_chebi"),
        'faiss_index_path': os.path.join(DATA_DIR, f"faiss_index_chebi.bin"),
        'faiss_metadata_path': os.path.join(DATA_DIR, f"faiss_metadata_chebi.json"),
    }
}
# Ensure Whoosh directories exist
for name, config_data in ONTOLOGIES_CONFIG.items():
    os.makedirs(config_data['whoosh_index_dir'], exist_ok=True)


# Output file from parse_ontology.py
ONTOLOGY_DUMP_JSON = os.path.join(DATA_DIR, "ontology_syns_dump.json")

# Output file for enriched documents
ENRICHED_DOCUMENTS_FILE = os.path.join(DATA_DIR, "enriched_documents.json")

# Output file for embeddings (used for building FAISS index)
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.json")

# Whoosh Index Directory
WHOOSH_INDEX_DIR = os.path.join(DATA_DIR, "whoosh_index")
os.makedirs(WHOOSH_INDEX_DIR, exist_ok=True)

# FAISS Index Paths
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
FAISS_METADATA_PATH = os.path.join(DATA_DIR, "faiss_metadata.json")

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
#EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" 
# RERANKER_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_BATCH_SIZE = 32 # For batch embedding script
EMBEDDING_DEVICE = 'cpu'  # or 'cuda' if available, for embedding script

# Default K values for retrieval
DEFAULT_K_LEXICAL = 20
DEFAULT_K_VECTOR = 20
DEFAULT_RERANK_K = DEFAULT_K_LEXICAL + DEFAULT_K_VECTOR 

# Namespaces (using string constants for broader compatibility if rdflib not always imported)
RDFS_NS_STR = "http://www.w3.org/2000/01/rdf-schema#"
RDF_NS_STR = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS_STR = "http://www.w3.org/2002/07/owl#"
SKOS_NS_STR = "http://www.w3.org/2004/02/skos/core#"
OBO_NS_STR = "http://purl.obolibrary.org/obo/"
OBOINOWL_NS_STR = "http://www.geneontology.org/formats/oboInOwl#"
IAO_NS_STR = "http://purl.obolibrary.org/obo/IAO_"

# For rdflib usage where Namespace objects are preferred:
RDFS_NS = Namespace(RDFS_NS_STR)
RDF_NS = Namespace(RDF_NS_STR)
OWL_NS = Namespace(OWL_NS_STR)
SKOS_NS = Namespace(SKOS_NS_STR)
OBO_NS = Namespace(OBO_NS_STR)
OBOINOWL_NS = Namespace(OBOINOWL_NS_STR)
IAO_NS = Namespace(IAO_NS_STR)


# Mapping of common relation URIs/CURIEs to human-readable names and default prefixes
# Used by parse_ontology.py and potentially enrich_documents.py
RELATION_CONFIG = {
    "obo:BFO_0000050": {"label": "part of", "prefix": "obo"},
    "obo:RO_0001000": {"label": "derives from", "prefix": "obo"},
    "obo:RO_0002200": {"label": "has phenotype", "prefix": "obo"},
    "obo:RO_0002451": {"label": "has part", "prefix": "obo"},
    "obo:FOODON_0000246": {"label": "has ingredient", "prefix": "obo"},
    "obo:RO_0000056": {"label": "participates in", "prefix": "obo"},
    "obo:RO_0000057": {"label": "has participant", "prefix": "obo"},
    "obo:RO_0000085": {"label": "functionally related to", "prefix": "obo"},
    "obo:RO_0002090": {"label": "adjacent to", "prefix": "obo"},
    "obo:RO_0002131": {"label": "overlaps", "prefix": "obo"},
    "obo:RO_0002150": {"label": "connected to", "prefix": "obo"},
    "obo:RO_0002202": {"label": "develops from", "prefix": "obo"},
    "obo:RO_0002350": {"label": "member of", "prefix": "obo"},
    "obo:RO_0002351": {"label": "has member", "prefix": "obo"},
    "obo:RO_0002353": {"label": "output of", "prefix": "obo"},
    "obo:RO_0002440": {"label": "causally related to", "prefix": "obo"},
    "obo:RO_0002450": {"label": "contains", "prefix": "obo"},
    "obo:RO_0002500": {"label": "surrounds", "prefix": "obo"},
    "obo:RO_0002573": {"label": "has quality", "prefix": "obo"},
    "obo:RO_0002606": {"label": "is transformation of", "prefix": "obo"},
    "obo:RO_0002607": {"label": "has input", "prefix": "obo"},
    "obo:RO_0003000": {"label": "used in", "prefix": "obo"},
    "obo:FOODON_0000380": {"label": "has preparation method", "prefix": "obo"},
    "obo:FOODON_0000226": {"label": "has food source", "prefix": "obo"},
    "obo:FOODON_0000388": {"label": "has maturity state", "prefix": "obo"},
    "obo:FOODON_0000286": {"label": "has packaging", "prefix": "obo"},
    "obo:FOODON_0000240": {"label": "has preservation method", "prefix": "obo"},
    "obo:FOODON_0000440": {"label": "has physical state", "prefix": "obo"},
    # "obo:FOODON_": {"label": "FoodON specific relation", "prefix": "obo"}, # This generic one might be too broad
    "obo:ERO_0000039": {"label": "has nutrient", "prefix": "obo"},
    "obo:ERO_0000589": {"label": "dietary context of", "prefix": "obo"},
    "obo:NCIT_C25277": {"label": "is allergen of", "prefix": "obo"},
    "obo:NCIT_C48670": {"label": "has active ingredient", "prefix": "obo"},
    "obo:NCIT_C64548": {"label": "part of human diet", "prefix": "obo"},
    "obo:PATO_0000001": {"label": "has quality PATO", "prefix": "obo"},
}

TARGET_RELATIONS_CURIES = list(RELATION_CONFIG.keys())

# CURIE Prefix Map for uri_to_curie and curie_to_uri conversions
# Ensure the keys are the *base URIs* that prefixes are expected for.
CURIE_PREFIX_MAP = {
    "http://purl.obolibrary.org/obo/FOODON_": "FOODON",
    "http://purl.obolibrary.org/obo/BFO_": "BFO",
    "http://purl.obolibrary.org/obo/RO_": "RO",
    "http://purl.obolibrary.org/obo/ERO_": "ERO",
    "http://purl.obolibrary.org/obo/NCIT_": "NCIT",
    "http://purl.obolibrary.org/obo/PATO_": "PATO",
    "http://purl.obolibrary.org/obo/IAO_": "IAO",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://www.geneontology.org/formats/oboInOwl#": "oboInOwl",
    "http://purl.obolibrary.org/obo/HANCESTRO_": "HANCESTRO",  
    "http://purl.obolibrary.org/obo/GAZ_": "GAZ",
    "http://purl.obolibrary.org/obo/CHEBI_": "CHEBI",
    "http://purl.obolibrary.org/obo/NCBITaxon_": "NCBITaxon",
    "http://purl.obolibrary.org/obo/UBERON_": "UBERON",
    "http://purl.obolibrary.org/obo/ENVO_": "ENVO",
    "http://purl.obolibrary.org/obo/HP_": "HP",
    "http://purl.obolibrary.org/obo/GO_": "GO",
    # General OBO prefix - should be last or handled carefully to avoid overly broad matches
    # if specific OBO sub-ontologies are listed above.
    "http://purl.obolibrary.org/obo/": "obo",
}

# LLM API Key (placeholders)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#LLM_SELECTOR_MODEL_NAME = "gemini-1.5-flash-latest"
LLM_SELECTOR_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
OLLAMA_SELECTOR_MODEL_NAME = 'llama3.1:8b'
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log") # Example log file in project root