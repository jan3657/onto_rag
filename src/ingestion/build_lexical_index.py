# src/ingestion/build_lexical_index.py
import sys
import os

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser
import traceback

# Now import using the 'src' package prefix
from src.config import ONTOLOGY_DUMP_PATH, WHOOSH_INDEX_DIR

# ... (rest of the build_lexical_index.py code, ensuring imports like `from src.config` are used)
def build_index(json_path: str, index_dir: str):
    print(f"Building lexical index from {json_path} into {index_dir}")

    schema = Schema(
        curie=ID(stored=True, unique=True),
        label=TEXT(stored=True, analyzer=None), # Keep None analyzer for exact matches if needed, or use default
        synonyms=TEXT(stored=True),
        definition=TEXT(stored=True),
        relations_text=TEXT(stored=False)
    )

    os.makedirs(index_dir, exist_ok=True)
    
    try:
        # create_in will overwrite if index exists. If you want to open, use open_dir.
        ix = create_in(index_dir, schema)
        print(f"Whoosh index schema created in {index_dir}")
    except Exception as e:
         print(f"Error creating Whoosh index directory or schema: {e}")
         traceback.print_exc()
         return

    writer = ix.writer()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)

        print(f"Indexing {len(ontology_data)} entities...")
        indexed_count = 0
        for curie, data in ontology_data.items():
            label = data.get('label', '') or '' # Ensure string
            synonyms_list = data.get('synonyms', [])
            synonyms = " ".join(s for s in synonyms_list if s) # Join non-empty synonyms

            definition = data.get('definition', '') or ''

            relations_text = ""
            relations_dict = data.get('relations', {})
            if relations_dict:
                 relations_text = " ".join([
                     f"{prop.replace('_', ' ')} {' '.join(targets)}" for prop, targets in relations_dict.items()
                 ])
            
            # Skip adding document if all text fields are empty (optional)
            # if not label and not synonyms and not definition and not relations_text:
            #     print(f"Skipping empty document for CURIE: {curie}")
            #     continue

            writer.add_document(
                curie=curie,
                label=label,
                synonyms=synonyms,
                definition=definition,
                relations_text=relations_text
            )
            indexed_count += 1
            if indexed_count % 1000 == 0:
                 print(f"Indexed {indexed_count} entities...")
        
        print(f"Committing index with {indexed_count} documents.")
        writer.commit()
        print("Lexical index built successfully.")

    except FileNotFoundError:
        print(f"Error: Ontology dump file not found at {json_path}")
        traceback.print_exc()
        writer.cancel() 
    except Exception as e:
        print(f"An error occurred during index building: {e}")
        traceback.print_exc()
        writer.cancel()

if __name__ == "__main__":
    if not os.path.exists(ONTOLOGY_DUMP_PATH):
        print(f"Error: Ontology dump file {ONTOLOGY_DUMP_PATH} not found.")
        print("Please run src/ingestion/parse_ontology.py first.")
    else:
        build_index(ONTOLOGY_DUMP_PATH, WHOOSH_INDEX_DIR)