# src/ingestion/build_lexical_index.py
import json
import whoosh.index
from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
import traceback
from pathlib import Path

from src.config import ONTOLOGIES_CONFIG

def build_single_index(json_path: Path, index_dir: Path):
    """Builds a single Whoosh index from a given JSON dump file."""
    print(f"Building lexical index from {json_path} into {index_dir}")

    schema = Schema(
        curie=ID(stored=True, unique=True),
        label=TEXT(stored=True),
        synonyms=TEXT(stored=True),
        definition=TEXT(stored=True),
        relations_text=TEXT(stored=False) # Indexed but not stored to save space
    )

    # Ensure the target directory exists
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        ix = create_in(str(index_dir), schema)
        print(f"Whoosh index schema created in {index_dir}")
    except (whoosh.index.LockError, whoosh.index.EmptyIndexError) as e:
         print(f"Error creating Whoosh index directory or schema: {e}")
         traceback.print_exc()
         return
         return

    writer = ix.writer()
    try:
        with json_path.open('r', encoding='utf-8') as f:
            ontology_data = json.load(f)

        print(f"Indexing {len(ontology_data)} entities...")
        indexed_count = 0
        for curie, data in ontology_data.items():
            label = data.get('label', '') or ''
            synonyms_list = data.get('synonyms', [])
            synonyms = " ".join(s for s in synonyms_list if s)
            definition = data.get('definition', '') or ''
            
            # Create a searchable string from relation data
            relations_text = ""
            relations_dict = data.get('relations', {})
            if relations_dict:
                 # The 'key' is the human-readable relation name (e.g., "has ingredient")
                 # The 'value' is a list of target CURIEs (which we don't need for this text field)
                 relations_text = " ".join(relations_dict.keys())
            
            writer.add_document(
                curie=curie,
                label=label,
                synonyms=synonyms,
                definition=definition,
                relations_text=relations_text
            )
            indexed_count += 1
            if indexed_count % 1000 == 0:
                 print(f"  ...indexed {indexed_count} entities...")
        
        print(f"Committing index with {indexed_count} documents.")
        writer.commit()
        print("Lexical index built successfully.")

    except FileNotFoundError:
        print(f"Error: Ontology dump file not found at {json_path}")
        traceback.print_exc()
        writer.cancel() 
    except (json.JSONDecodeError, whoosh.writing.IndexingError, IOError) as e:
        print(f"An error occurred during index building: {e}")
        traceback.print_exc()
        writer.cancel()

def main():
    """Loops over all configured ontologies and builds a lexical index for each."""
    for name, config_data in ONTOLOGIES_CONFIG.items():
        dump_path = config_data.get('dump_json_path')
        index_dir = config_data.get('whoosh_index_dir')
        
        print(f"\n--- Building Lexical Index for: '{name}' ---")

        if not dump_path or not index_dir:
            print(f"Warning: Configuration for '{name}' is missing 'dump_json_path' or 'whoosh_index_dir'. Skipping.")
            continue
            
        if not dump_path.exists():
            print(f"Error: Ontology dump file not found at {dump_path}. Skipping '{name}'.")
            print("Please run 'src/ingestion/parse_ontology.py' and 'src/ingestion/enrich_documents.py' first.")
            continue
            
        build_single_index(dump_path, index_dir)
        
    print("\n--- All lexical indexes have been built. ---")

if __name__ == "__main__":
    main()