# src/ingestion/build_whoosh_index.py
"""
Build a Whoosh lexical search index from an ontology dump JSON.

The index enables fast full-text search across term labels, synonyms,
definitions, and relation descriptions.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from whoosh import index as whoosh_index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer

from src import config

logger = logging.getLogger(__name__)


def build_whoosh_index(
    ontology_dump_path: Path,
    index_dir: Path,
    fields: Optional[List[str]] = None,
    store_label: bool = False,
) -> None:
    """
    Build a Whoosh lexical search index from an ontology dump.

    Parameters
    ----------
    ontology_dump_path : Path
        Path to the ontology dump JSON file (output of parse_ontology).
    index_dir : Path
        Directory where the Whoosh index will be created.
    fields : Optional[List[str]]
        Fields to index for search.
        Default: config.WHOOSH_FIELDS
    store_label : bool
        Whether to store label text in Whoosh stored fields.
        Keep False for very large ontologies to avoid 4GB stored-column overflow.
    """
    # Apply defaults from config
    fields = fields or config.WHOOSH_FIELDS

    logger.info(f"Building Whoosh index from: {ontology_dump_path}")
    logger.info(f"Index directory: {index_dir}")
    logger.info(f"Indexing fields: {fields}")
    logger.info(f"Store label in index: {store_label}")

    if not ontology_dump_path.exists():
        raise FileNotFoundError(f"Ontology dump not found: {ontology_dump_path}")

    # Load ontology dump
    with ontology_dump_path.open("r", encoding="utf-8") as f:
        ontology_data: Dict[str, Any] = json.load(f)

    logger.info(f"Loaded {len(ontology_data)} terms from dump")

    # Rebuild index from scratch to avoid stale locks/temp files from interrupted runs
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Define schema based on requested fields
    # Always include curie (ID) and label (indexed text).
    # On huge ontologies, storing labels can overflow Whoosh's 32-bit byte offsets.
    analyzer = StemmingAnalyzer()

    schema_fields = {
        "curie": ID(stored=True, unique=True),
        "label": TEXT(stored=store_label, analyzer=analyzer),
    }

    # Add optional searchable fields
    # NOTE: Large fields are searchable but NOT stored to avoid overflow on huge datasets (64M+ entries)
    # The retriever loads full term details from the JSON dump anyway, so we only need the CURIE
    if "synonyms" in fields:
        schema_fields["synonyms"] = TEXT(stored=False, analyzer=analyzer)
    if "definition" in fields:
        schema_fields["definition"] = TEXT(stored=False, analyzer=analyzer)
    if "relations_text" in fields:
        schema_fields["relations_text"] = TEXT(stored=False, analyzer=analyzer)

    schema = Schema(**schema_fields)

    # Create or overwrite the index
    ix = whoosh_index.create_in(str(index_dir), schema)
    writer = ix.writer()

    indexed_count = 0
    for curie, term_data in ontology_data.items():
        label = term_data.get("label", "")
        if not label:
            continue

        doc = {
            "curie": curie,
            "label": label,
        }

        if "synonyms" in fields:
            synonyms = term_data.get("synonyms", [])
            doc["synonyms"] = " ; ".join(synonyms) if synonyms else ""

        if "definition" in fields:
            doc["definition"] = term_data.get("definition", "")

        if "relations_text" in fields:
            doc["relations_text"] = term_data.get("relations_text", "")

        writer.add_document(**doc)
        indexed_count += 1

    writer.commit()
    logger.info(f"Indexed {indexed_count} terms to Whoosh")
    logger.info(f"Whoosh index saved to: {index_dir}")


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.build_whoosh_index <dump.json> <index_dir>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    build_whoosh_index(Path(sys.argv[1]), Path(sys.argv[2]))
