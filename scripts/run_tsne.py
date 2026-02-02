#!/usr/bin/env python3
"""
Compute sentence-transformer embeddings for queries and project them with t-SNE.

Outputs:
- data/embeddings/<encoder_slug>_tsne.csv   (coords + metadata)
- data/embeddings/<encoder_slug>_tsne.html  (interactive Plotly scatter)

Usage:
  .venv/bin/python scripts/run_tsne.py --only-incorrect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def slugify_encoder(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def load_errors(path: Path, only_incorrect: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if only_incorrect:
        df = df[df["is_correct"] == False]  # noqa: E712
    return df.reset_index(drop=True)


def compute_embeddings(texts: Iterable[str], encoder_name: str, batch_size: int = 64):
    model = SentenceTransformer(encoder_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def build_tsne_features(df: pd.DataFrame, encoder_name: str):
    # Deduplicate texts to save time
    unique_texts = sorted(set(df["query"].astype(str).tolist()))
    text_to_idx: Dict[str, int] = {t: i for i, t in enumerate(unique_texts)}

    print(f"Encoding {len(unique_texts)} unique queries with {encoder_name} ...")
    emb_unique = compute_embeddings(unique_texts, encoder_name)

    # Map back to full dataframe order
    emb_full = emb_unique[[text_to_idx[t] for t in df["query"].astype(str)]]
    return emb_full


def reduce_tsne(vectors, perplexity: int, pca_components: int):
    data = vectors
    if pca_components and pca_components < vectors.shape[1]:
        print(f"Running TruncatedSVD to {pca_components} dims before t-SNE ...")
        svd = TruncatedSVD(n_components=pca_components, random_state=42)
        data = svd.fit_transform(vectors)

    n = data.shape[0]
    perp = min(perplexity, max(5, n - 1))
    print(f"Running t-SNE on {n} points (perplexity={perp}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        metric="cosine",
        init="pca",
        random_state=42,
        learning_rate="auto",
        max_iter=1500,
        verbose=1,
    )
    coords = tsne.fit_transform(data)
    return coords


def make_plot(df: pd.DataFrame, html_path: Path):
    fig = px.scatter(
        df,
        x="tsne_x",
        y="tsne_y",
        color="error_type",
        symbol="dataset",
        hover_data={
            "query": True,
            "predicted_label": True,
            "gold_ids": True,
            "model": True,
            "run_id": True,
            "confidence": True,
            "is_correct": True,
            "tsne_x": False,
            "tsne_y": False,
        },
        title="t-SNE of queries (colored by error_type)",
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote interactive plot to {html_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", type=Path, default=Path("data/analysis/errors.parquet"))
    ap.add_argument("--encoder", type=str, default="intfloat/e5-small-v2")
    ap.add_argument("--only-incorrect", action="store_true", help="Focus on incorrect rows only")
    ap.add_argument("--perplexity", type=int, default=35)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--outdir", type=Path, default=Path("data/embeddings"))
    args = ap.parse_args()

    df = load_errors(args.errors, args.only_incorrect)
    if df.empty:
        raise SystemExit("No rows found after filtering.")

    encoder_slug = slugify_encoder(args.encoder)
    coords_csv = args.outdir / f"{encoder_slug}_tsne.csv"
    html_path = args.outdir / f"{encoder_slug}_tsne.html"

    vectors = build_tsne_features(df, args.encoder)
    coords = reduce_tsne(vectors, args.perplexity, args.pca_components)

    df = df.copy()
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]

    coords_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(coords_csv, index=False)
    print(f"Wrote coords CSV to {coords_csv}")

    make_plot(df, html_path)


if __name__ == "__main__":
    main()
