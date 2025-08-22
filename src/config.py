# src/config.py
from pathlib import Path
from dotenv import load_dotenv
from rdflib import Namespace
from os import getenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Path Configuration (using pathlib) ---

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env file in the project root
load_dotenv(PROJECT_ROOT / ".env")

# Data Directory (for ontology dump, indexes, etc.)
DATA_DIR = PROJECT_ROOT / "data"

# Cache Path
PIPELINE_CACHE_PATH = DATA_DIR / "pipeline_cache.json"

# Ontologies Directory
ONTOLOGIES_DIR = PROJECT_ROOT / "ontologies"

# --- Ontology Configuration ---
ONTOLOGIES_CONFIG = {
    'foodon': {
        'path': ONTOLOGIES_DIR / "foodon.owl",
        'prefix': 'FOODON:',
        'id_pattern': r'^FOODON:\d+$',
        'dump_json_path': DATA_DIR / "ontology_dump_foodon.json",
        'enriched_docs_path': DATA_DIR / "enriched_documents_foodon.json",
        'embeddings_path': DATA_DIR / "embeddings_foodon.json",
        'whoosh_index_dir': DATA_DIR / "whoosh_index_foodon",
        'faiss_index_path': DATA_DIR / "faiss_index_foodon.bin",
        'faiss_metadata_path': DATA_DIR / "faiss_metadata_foodon.json",
    },
    'chebi': {
        'path': ONTOLOGIES_DIR / "chebi.owl",
        'prefix': 'CHEBI:',
        'id_pattern': r'^CHEBI:\d+$',
        'dump_json_path': DATA_DIR / "ontology_dump_chebi.json",
        'enriched_docs_path': DATA_DIR / "enriched_documents_chebi.json",
        'embeddings_path': DATA_DIR / "embeddings_chebi.json",
        'whoosh_index_dir': DATA_DIR / "whoosh_index_chebi",
        'faiss_index_path': DATA_DIR / "faiss_index_chebi.bin",
        'faiss_metadata_path': DATA_DIR / "faiss_metadata_chebi.json",
    }
}
# NOTE: The loop that created Whoosh directories has been removed.
# The script responsible for building the Whoosh index should create its own directory.

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = 'cpu'

# --- Retrieval and Reranking Configuration ---
DEFAULT_K_LEXICAL = 15
DEFAULT_K_VECTOR = 15
DEFAULT_RERANK_K = DEFAULT_K_LEXICAL + DEFAULT_K_VECTOR

# --- Namespace Configuration ---
RDFS_NS_STR = "http://www.w3.org/2000/01/rdf-schema#"
RDF_NS_STR = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS_STR = "http://www.w3.org/2002/07/owl#"
SKOS_NS_STR = "http://www.w3.org/2004/02/skos/core#"
OBO_NS_STR = "http://purl.obolibrary.org/obo/"
OBOINOWL_NS_STR = "http://www.geneontology.org/formats/oboInOwl#"
IAO_NS_STR = "http://purl.obolibrary.org/obo/IAO_"

RDFS_NS = Namespace(RDFS_NS_STR)
RDF_NS = Namespace(RDF_NS_STR)
OWL_NS = Namespace(OWL_NS_STR)
SKOS_NS = Namespace(SKOS_NS_STR)
OBO_NS = Namespace(OBO_NS_STR)
OBOINOWL_NS = Namespace(OBOINOWL_NS_STR)
IAO_NS = Namespace(IAO_NS_STR)


# Mapping of common relation URIs/CURIEs to human-readable names and default prefixes, used by parse_ontology.py and potentially enrich_documents.py
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
    "obo:ERO_0000039": {"label": "has nutrient", "prefix": "obo"},
    "obo:ERO_0000589": {"label": "dietary context of", "prefix": "obo"},
    "obo:NCIT_C25277": {"label": "is allergen of", "prefix": "obo"},
    "obo:NCIT_C48670": {"label": "has active ingredient", "prefix": "obo"},
    "obo:NCIT_C64548": {"label": "part of human diet", "prefix": "obo"},
    "obo:PATO_0000001": {"label": "has quality PATO", "prefix": "obo"},
}

TARGET_RELATIONS_CURIES = list(RELATION_CONFIG.keys())

# CURIE Prefix Map for uri_to_curie and curie_to_uri conversions
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
    "http://purl.obolibrary.org/obo/": "obo",
}

# --- Pipeline Loop Configuration ---
CONFIDENCE_THRESHOLD = 0.6  # If score is below this, try to generate synonyms
MAX_PIPELINE_LOOPS = 4     # Max number of attempts (initial + retries)
# Max number of concurrent LLM API calls for async processing
MAX_CONCURRENT_REQUESTS = 50


GEMINI_API_KEY = getenv("GEMINI_API_KEY")
GEMINI_SELECTOR_MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_SCORER_MODEL_NAME = "gemini-2.5-pro"  # "gemini-2.5-pro"
GEMINI_SYNONYM_MODEL_NAME = "gemini-2.5-flash-lite"

OLLAMA_SELECTOR_MODEL_NAME = 'llama3.1:8b'
OLLAMA_SCORER_MODEL_NAME = 'llama3.1:8b'
OLLAMA_SYNONYM_MODEL_NAME = 'llama3.1:8b'

HF_SELECTOR_MODEL_ID = "arcee-ai/AFM-4.5B"
HF_MODEL_KWARGS = {
    "torch_dtype": torch.bfloat16,
    "device_map": "mps",
}
HF_GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "top_k": 50
}

# Path to the prompt template for the selector
SELECTOR_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / \
    "chebi_selection.tpl" #"final_selection.tpl"  # "strict_selection_minimal.tpl"
CONFIDENCE_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / \
    "chebi_confidence.tpl" #"confidence_assessment3.tpl"  # "confidence_assessment.tpl"
SYNONYM_PROMPT_TEMPLATE_PATH = PROJECT_ROOT /"prompts" / \
    "chebi_synonyms.tpl" #"synonym_generation.tpl"

PIPELINE = "gemini"  # "gemini", "ollama", or "huggingface"

# Logging configuration
LOG_LEVEL = "WARNING"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = PROJECT_ROOT / "app.log"
