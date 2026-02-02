# src/ingestion/__init__.py
"""
Modular ingestion pipeline for ontology processing.

Functions read defaults from config.py but accept optional parameters
for per-script overrides.

Usage:
    from src.ingestion import parse_ontology, build_whoosh_index, build_embeddings, build_faiss_index
    
    # Uses config defaults
    parse_ontology(ontology_path, output_path)
    
    # Override specific parameters
    build_embeddings(dump_path, output_path, model_name="custom-model")
    
    # For TSV-based knowledge bases
    from src.ingestion import parse_tsv
    parse_tsv(tsv_path, output_path, id_column="GeneID", label_column="Symbol")
"""

from src.ingestion.parse_ontology import parse_ontology
from src.ingestion.parse_tsv import parse_tsv
from src.ingestion.build_whoosh_index import build_whoosh_index
from src.ingestion.build_embeddings import build_embeddings, build_faiss_index

__all__ = [
    "parse_ontology",
    "parse_tsv",
    "build_whoosh_index", 
    "build_embeddings",
    "build_faiss_index",
]
