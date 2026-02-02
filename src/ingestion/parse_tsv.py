# src/ingestion/parse_tsv.py
"""
Generic TSV ontology/knowledge base parser.

Converts tabular data (TSV/CSV) into the standard JSON dump format
used by the indexing pipeline.

Each evaluation script specifies the column mapping for its data source.
"""

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src import config

logger = logging.getLogger(__name__)


def parse_tsv(
    tsv_path: Path,
    output_path: Path,
    id_column: str,
    label_column: str,
    id_prefix: str = "",
    synonyms_column: Optional[str] = None,
    synonyms_separator: str = "|",
    definition_column: Optional[str] = None,
    filter_func: Optional[Callable[[Dict[str, str]], bool]] = None,
    transform_func: Optional[Callable[[Dict[str, str]], Dict[str, Any]]] = None,
    delimiter: str = "\t",
    skip_rows_starting_with: str = "#",
    header_starts_with: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse a TSV/CSV file into ontology dump JSON format.
    
    Parameters
    ----------
    tsv_path : Path
        Path to the TSV/CSV file.
    output_path : Path
        Path where the JSON dump will be saved.
    id_column : str
        Column name for ID.
    label_column : str
        Column name for Label.
    id_prefix : str
        Prefix to add to IDs.
    synonyms_column : Optional[str]
        Column name for synonyms.
    synonyms_separator : str
        Separator for synonyms.
    definition_column : Optional[str]
        Column name for definition.
    filter_func : Optional[Callable[[Dict[str, str]], bool]]
        Row filter function.
    transform_func : Optional[Callable[[Dict[str, str]], Dict[str, Any]]]
        Row transform function.
    delimiter : str
        Field delimiter.
    skip_rows_starting_with : str
        Skip lines starting with this (comments). Ignored if header_starts_with is found.
    header_starts_with : Optional[str]
        If provided, scans the file until a line starting with this string is found, using it as the header.
    max_rows : Optional[int]
        Max rows to process.
    """
    logger.info(f"Parsing TSV from: {tsv_path}")
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Open file (handle gzip)
    if str(tsv_path).endswith('.gz'):
        file_handle = gzip.open(tsv_path, 'rt', encoding='utf-8')
    else:
        file_handle = tsv_path.open('r', encoding='utf-8')
    
    ontology_data: Dict[str, Any] = {}
    processed = 0
    skipped = 0
    
    try:
        # Header detection logic
        if header_starts_with:
            # excessive scanning protection
            found_header = False
            for line in file_handle:
                if line.startswith(header_starts_with):
                    found_header = True
                    # This line is the header.
                    # We need to pass it to DictReader.
                    # Use a generator that yields this line first, then the rest of the file
                    from itertools import chain
                    
                    # Create a fresh iterator from the current position
                    # But DictReader needs the header line to be the first yielded item
                    # Since we consumed it, we reconstruct the stream
                    reader_lines = chain([line], file_handle)
                    reader = csv.DictReader(reader_lines, delimiter=delimiter)
                    break
            
            if not found_header:
                raise ValueError(f"Header starting with '{header_starts_with}' not found in file")
        
        else:
            # Fallback: Assume first line is header (or handle comments if basic logic needed)
            # For strict control, users should use header_starts_with
            reader = csv.DictReader(file_handle, delimiter=delimiter)

        # Clean field names: remove leading '#' and whitespace
        if reader.fieldnames:
            new_fieldnames = []
            for name in reader.fieldnames:
                # Remove leading '#' (only if it's the start, often first col)
                clean_name = name.lstrip('#')
                # Strip whitespace
                clean_name = clean_name.strip()
                new_fieldnames.append(clean_name)
            reader.fieldnames = new_fieldnames
        
        logger.info(f"TSV columns: {reader.fieldnames[:5]}...")  # Log first 5 columns
        
        for row in reader:
            # Skip comment lines if they occur in data body (and aren't the header we just found)
            # Checking values of the first column can be a heuristic
            first_val = list(row.values())[0] if row else ""
            if first_val and first_val.startswith(skip_rows_starting_with):
                continue
                skipped += 1
                continue
            
            # Get ID
            raw_id = row.get(id_column, "").strip()
            if not raw_id or raw_id == "-":
                skipped += 1
                continue
            
            entity_id = f"{id_prefix}{raw_id}"
            
            # Get label
            label = row.get(label_column, "").strip()
            if not label or label == "-":
                skipped += 1
                continue
            
            # Use custom transform if provided
            if transform_func:
                entry = transform_func(row)
                entry.setdefault("label", label)
            else:
                # Default extraction
                synonyms: List[str] = []
                if synonyms_column:
                    syn_str = row.get(synonyms_column, "").strip()
                    if syn_str and syn_str != "-":
                        synonyms = [s.strip() for s in syn_str.split(synonyms_separator) if s.strip()]
                
                definition = ""
                if definition_column:
                    definition = row.get(definition_column, "").strip()
                    if definition == "-":
                        definition = ""
                
                entry = {
                    "label": label,
                    "synonyms": synonyms,
                    "definition": definition,
                    "parents": [],
                    "relations": [],
                    "relations_text": "",
                }
            
            ontology_data[entity_id] = entry
            processed += 1
            
            if processed % 100000 == 0:
                logger.info(f"Processed {processed:,} entries...")
            
            if max_rows and processed >= max_rows:
                logger.info(f"Reached max_rows limit ({max_rows})")
                break
    
    finally:
        file_handle.close()
    
    logger.info(f"Processed {processed:,} entries, skipped {skipped:,}")
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ontology_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved ontology dump to: {output_path}")
    return ontology_data


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 5:
        print("Usage: python -m src.ingestion.parse_tsv <input.tsv> <output.json> <id_col> <label_col>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    parse_tsv(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        id_column=sys.argv[3],
        label_column=sys.argv[4],
    )
