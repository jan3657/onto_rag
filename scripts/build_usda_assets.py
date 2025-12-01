"""
Builds USDA assets for OntoRAG-style retrieval:
- usda_dump.json (id -> label/synonyms/definition)
- usda_enriched.json (list of {id,label,text})
- usda_embeddings.json
- Whoosh index + FAISS index
"""

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    ONTOLOGIES_CONFIG,
)
from src.ingestion.build_lexical_index import build_single_index
from src.infrastructure.retrieval.faiss_store import FAISSVectorStore


USDA_CFG = ONTOLOGIES_CONFIG["usda"]
CSV_PATH = Path("USA_AU_data/processed/usda_complete.csv")
DUMP_PATH = USDA_CFG["dump_json_path"]
ENRICHED_PATH = USDA_CFG["enriched_docs_path"]
EMBED_PATH = USDA_CFG["embeddings_path"]
WHOOSH_DIR = USDA_CFG["whoosh_index_dir"]
FAISS_INDEX = USDA_CFG["faiss_index_path"]
FAISS_META = USDA_CFG["faiss_metadata_path"]

KEY_NUTRIENTS = [
    "prot_g",
    "fat_g",
    "fasat_g",
    "sugar_g",
    "fibt_g",
    "na_mg",
    "fe_mg",
    "p_mg",
    "enerakcal",
    "enerakj",
]


def load_usda() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # ensure id prefix
    df["id"] = df["id"].apply(lambda x: f"USDA:{x}")
    return df


def make_dump(df: pd.DataFrame) -> Dict[str, Dict]:
    dump = {}
    for _, row in df.iterrows():
        curie = row["id"]
        name = row.get("name", "")
        classification = row.get("classification", "")
        synonyms: List[str] = []
        if isinstance(classification, str) and classification.strip():
            synonyms.append(classification.strip())

        # Build a terse definition from classification + top nutrients
        nutrient_bits = []
        for n in KEY_NUTRIENTS:
            if n in row and pd.notna(row[n]):
                nutrient_bits.append(f"{n}={row[n]}")
        definition = f"Classification: {classification}. Nutrients: " + "; ".join(nutrient_bits)

        dump[curie] = {
            "label": name,
            "synonyms": synonyms,
            "definition": definition,
            "relations": {},
        }
    return dump


def make_enriched(df: pd.DataFrame) -> List[Dict]:
    enriched = []
    for _, row in df.iterrows():
        name = row.get("name", "")
        classification = row.get("classification", "")
        nutrient_bits = []
        for n in KEY_NUTRIENTS:
            if n in row and pd.notna(row[n]):
                nutrient_bits.append(f"{n}={row[n]}")
        nutrient_str = "; ".join(nutrient_bits)
        text = f"Name: {name}. Classification: {classification}. Nutrients: {nutrient_str}"
        enriched.append({"id": row["id"], "label": name, "text": text})
    return enriched


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_embeddings(enriched: List[Dict]):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    texts = [item["text"] for item in enriched]
    ids = [item["id"] for item in enriched]
    labels = [item["label"] for item in enriched]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb_out = []
    for i, emb in enumerate(embeddings):
        emb_out.append({"id": ids[i], "label": labels[i], "embedding": emb.tolist()})
    save_json(emb_out, EMBED_PATH)
    return emb_out


def build_faiss(embeddings: List[Dict]):
    store = FAISSVectorStore(
        index_path=FAISS_INDEX,
        metadata_path=FAISS_META,
        embeddings_file_path=EMBED_PATH,
    )
    # store initialised itself; ensure saved
    if store.index is None or not store.metadata:
        raise RuntimeError("FAISS store failed to build; check embeddings.")


def main():
    df = load_usda()
    dump = make_dump(df)
    save_json(dump, DUMP_PATH)
    print(f"Wrote dump: {DUMP_PATH} ({len(dump)} rows)")

    enriched = make_enriched(df)
    save_json(enriched, ENRICHED_PATH)
    print(f"Wrote enriched docs: {ENRICHED_PATH} ({len(enriched)} rows)")

    embeddings = build_embeddings(enriched)
    print(f"Wrote embeddings: {EMBED_PATH}")

    build_faiss(embeddings)
    print(f"FAISS index ready at {FAISS_INDEX}")

    build_single_index(DUMP_PATH, WHOOSH_DIR)
    print(f"Whoosh index ready at {WHOOSH_DIR}")


if __name__ == "__main__":
    main()
