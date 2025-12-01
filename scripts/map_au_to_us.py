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
from typing import Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as asyncio_tqdm

from src.adapters.pipeline_factory import create_pipeline
from src import config
from src.utils.logging_config import setup_run_logging
from src.utils.token_tracker import token_tracker

setup_run_logging("au_to_us")
logger = logging.getLogger("au_to_us")

AU_CSV = Path("USA_AU_data/processed/ausnut_complete.csv")
OUT_CSV = Path("USA_AU_data/processed/au_us_matches_rag.csv")
CACHE_PATH = Path("USA_AU_data/processed/au_us_rag_cache.json")

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
    "enerckcal",
    "enerckj",
]


def build_context(row: pd.Series) -> str:
    parts: List[str] = []
    classification = row.get("classification", "")
    if isinstance(classification, str) and classification.strip():
        parts.append(f"Classification: {classification}")
    nutrients = []
    for n in CONTEXT_NUTRIENTS:
        if n in row and pd.notna(row[n]):
            nutrients.append(f"{n}={row[n]}")
    if nutrients:
        parts.append("Nutrients: " + "; ".join(nutrients))
    return ". ".join(parts)


async def map_row(row: pd.Series, pipeline, semaphore: asyncio.Semaphore) -> Dict[str, Optional[str]]:
    query = row["name"]
    context = build_context(row)
    result, _cands = await pipeline.run(query=query, context=context, semaphore=semaphore)
    return {
        "au_id": row["id"],
        "au_name": query,
        "us_id": result.get("id") if result else None,
        "us_name": result.get("label") if result else None,
        "confidence_score": result.get("confidence_score") if result else None,
        "selector_explanation": result.get("selector_explanation") if result else None,
        "scorer_explanation": result.get("scorer_explanation") if result else None,
    }


async def main(limit: Optional[int] = None):
    df = pd.read_csv(AU_CSV)
    if limit:
        df = df.head(limit)

    pipeline = create_pipeline(config.PIPELINE)
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    try:
        tasks = [map_row(row, pipeline, semaphore) for _, row in df.iterrows()]
        results: List[Dict] = await asyncio_tqdm.gather(*tasks, desc="AU->US matching")
    finally:
        pipeline.close()
        logger.info(token_tracker.report_usage())

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} mappings to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map AUSNUT foods to USDA using OntoRAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of AUSNUT rows for a quick run.")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
