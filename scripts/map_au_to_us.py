"""
Map AUSNUT rows to USDA catalogue using the OntoRAG pipeline.

Inputs:
- USA_AU_data/processed/ausnut_complete.csv
- USDA assets (dump/enriched/embeddings/indexes) built via scripts/build_usda_assets.py

Output:
- USA_AU_data/processed/au_us_matches_rag.csv
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as asyncio_tqdm

from src.adapters.pipeline_factory import create_pipeline
from src import config
from src.utils.logging_config import setup_run_logging
from src.utils.token_tracker import token_tracker

setup_run_logging("au_to_us")
logger = logging.getLogger("au_to_us")

AU_CSV = Path("USA_AU_data/processed/ausnut_complete.csv")
OUT_CSV = Path("USA_AU_data/processed/au_us_matches_rag_top3.csv")
CACHE_PATH = Path("USA_AU_data/processed/au_us_rag_cache.json")
US_CSV = Path("USA_AU_data/processed/usda_complete.csv")
AU_ID_FILE = Path("USA_AU_data/au_ids.txt")

# Nutrients to include in context
CONTEXT_NUTRIENTS = [
    "prot_g",
    "fat_g",
    "fasat_g",
    "sugar_g",
    "fibt_g",
    "na_mg",
    "fe_mg",
    "p_mg",
    "energy_kj",  # normalized: AU uses enerckj, US uses enerakj
]


def build_context(row: pd.Series) -> str:
    parts: List[str] = []
    classification = row.get("classification", "")
    if isinstance(classification, str) and classification.strip():
        parts.append(f"Classification: {classification}")
    nutrients = []
    # Energy (kJ)
    energy_kj = row.get("enerckj", None)
    if energy_kj is None:
        energy_kj = row.get("enerakj", None)
    if energy_kj is not None and pd.notna(energy_kj):
        nutrients.append(f"energy_kj={energy_kj}")
    # Other nutrients
    for n in CONTEXT_NUTRIENTS:
        if n == "energy_kj":
            continue
        if n in row and pd.notna(row[n]):
            nutrients.append(f"{n}={row[n]}")
    if nutrients:
        parts.append("Nutrients: " + "; ".join(nutrients))
    return ". ".join(parts)


def extract_attrs(row: Optional[pd.Series], prefix: str) -> Dict[str, Any]:
    """Pull comparable attributes for easier human inspection."""
    if row is None:
        return {f"{prefix}_{field}": None for field in ["id", "name", "classification", *CONTEXT_NUTRIENTS]}
    out = {
        f"{prefix}_id": row.get("id"),
        f"{prefix}_name": row.get("name"),
        f"{prefix}_classification": row.get("classification"),
    }
    # Energy (kJ): AU column = enerckj, US column = enerakj
    energy_kj = row.get("enerckj", None)
    if energy_kj is None:
        energy_kj = row.get("enerakj", None)
    out[f"{prefix}_energy_kj"] = energy_kj

    for n in CONTEXT_NUTRIENTS:
        if n == "energy_kj":
            continue
        out[f"{prefix}_{n}"] = row.get(n)
    return out


def load_au_id_whitelist() -> List[str]:
    if AU_ID_FILE.exists():
        ids = [
            line.strip()
            for line in AU_ID_FILE.read_text().splitlines()
            if line.strip()
        ]
        return ids
    return []


async def map_row(row: pd.Series, us_lookup: Dict[str, pd.Series], pipeline, semaphore: asyncio.Semaphore) -> List[Dict[str, Optional[str]]]:
    query = row["name"]
    context = build_context(row)
    run_out = await pipeline.run(query=query, context=context, semaphore=semaphore)
    results, _cands = run_out if run_out else ([], [])
    mapped_rows: List[Dict[str, Optional[str]]] = []

    if not results:
        mapped_rows.append({
            "au_id": row["id"],
            "au_name": query,
            "rank": None,
            "us_id": None,
            "us_name": None,
            "confidence_score": None,
            "selector_confidence": None,
            "selector_explanation": "Selector returned no valid selection.",
            "scorer_explanation": None,
            **extract_attrs(row, "au"),
            **extract_attrs(None, "us"),
        })
        return mapped_rows

    for rank, result in enumerate(results, start=1):
        us_row = us_lookup.get(result.get("id")) if result else None
        mapped_rows.append({
            "au_id": row["id"],
            "au_name": query,
            "rank": rank,
            "us_id": result.get("id") if result else None,
            "us_name": result.get("label") if result else None,
            "confidence_score": result.get("confidence_score") if result else None,
            "selector_confidence": result.get("selector_confidence"),
            "selector_explanation": result.get("selector_explanation") if result else None,
            "scorer_explanation": result.get("scorer_explanation") if result else None,
            **extract_attrs(row, "au"),
            **extract_attrs(us_row, "us"),
        })
    return mapped_rows


async def main(limit: Optional[int] = None):
    df = pd.read_csv(AU_CSV)
    # Whitelist file takes priority over random sampling
    id_whitelist = load_au_id_whitelist()
    if id_whitelist:
        before = len(df)
        df = df[df["id"].astype(str).isin(id_whitelist)]
        missing = set(id_whitelist) - set(df["id"].astype(str))
        logger.info(f"Whitelist provided ({len(id_whitelist)} IDs). Matched {len(df)}/{before} AU rows.")
        if missing:
            logger.warning(f"{len(missing)} IDs from whitelist not found in AUSNUT data. Example: {list(missing)[:10]}")
        if df.empty:
            logger.error("After applying whitelist, no AUSNUT rows remain. Check ID format (AUSNUT IDs are alphanumeric, e.g., '10E10098').")
            return

        # Allow limit to further reduce the whitelist set
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42)
            logger.info(f"Applying limit={limit} to whitelisted IDs; running {len(df)} rows.")
    elif limit:
        df = df.sample(n=limit, random_state=42)

    # Load USDA rows and prefix IDs to match retriever outputs
    us_df = pd.read_csv(US_CSV)
    us_df["id"] = us_df["id"].apply(lambda x: f"USDA:{x}")
    us_lookup = {row["id"]: row for _, row in us_df.iterrows()}

    pipeline = create_pipeline(config.PIPELINE)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    try:
        tasks = [map_row(row, us_lookup, pipeline, semaphore) for _, row in df.iterrows()]
        per_row_results: List[List[Dict]] = await asyncio_tqdm.gather(*tasks, desc="AU->US matching")
    finally:
        pipeline.close()
        logger.info(token_tracker.report_usage())

    flattened: List[Dict] = [item for sublist in per_row_results for item in sublist]
    out_df = pd.DataFrame(flattened)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} mappings to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map AUSNUT foods to USDA using OntoRAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of AUSNUT rows for a quick run.")
    parser.add_argument("--au-id-file", type=Path, default=AU_ID_FILE, help="Path to file with AU IDs (one per line). Overrides --limit if present.")
    args = parser.parse_args()
    AU_ID_FILE = args.au_id_file
    asyncio.run(main(limit=args.limit))
