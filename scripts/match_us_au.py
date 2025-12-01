"""
Quick heuristic matcher between AUSNUT and USDA food rows.

Scoring:
- Name similarity via SequenceMatcher on lowercased, punctuation-stripped names.
- Nutrient similarity via cosine on z-scored shared numeric columns.
- Combined score = 0.6 * name + 0.4 * nutrient (fallbacks to name only if nutrients missing).

Outputs:
- USA_AU_data/processed/au_us_matches.csv with top match per AU row.
"""

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("USA_AU_data")
PROCESSED_DIR = DATA_DIR / "processed"
AU_PATH = PROCESSED_DIR / "ausnut_complete.csv"
US_PATH = PROCESSED_DIR / "usda_complete.csv"
OUT_PATH = PROCESSED_DIR / "au_us_matches.csv"


def normalize_name(name: str) -> str:
    """Lowercase and strip punctuation/extra spaces for stable comparison."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def name_similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio between two normalized names."""
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def pick_common_numeric_columns(df_au: pd.DataFrame, df_us: pd.DataFrame) -> List[str]:
    """Common numeric column names between AU and US frames."""
    numeric_au = {c for c in df_au.columns if pd.api.types.is_numeric_dtype(df_au[c])}
    numeric_us = {c for c in df_us.columns if pd.api.types.is_numeric_dtype(df_us[c])}
    return sorted(numeric_au & numeric_us)


def standardize_matrix(df: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Return matrix with z-scored columns (dropping all-NaN/zero-variance columns)."""
    cleaned_cols: List[str] = []
    matrices: List[np.ndarray] = []
    for col in columns:
        series = df[col]
        if series.isna().all():
            continue
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            continue
        cleaned_cols.append(col)
        matrices.append(((series - mean) / std).fillna(0.0).to_numpy())
    if not matrices:
        return np.empty((len(df), 0)), []
    return np.column_stack(matrices), cleaned_cols


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Return row-normalized matrix with NaNs replaced by zeros."""
    if mat.shape[1] == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_matches(df_au: pd.DataFrame, df_us: pd.DataFrame, weight_name: float = 0.6, nutrient_top_k: int = 50, token_top_k: int = 150) -> pd.DataFrame:
    common_cols = pick_common_numeric_columns(df_au, df_us)
    mat_au_raw, used_cols = standardize_matrix(df_au, common_cols)
    mat_us_raw, _ = standardize_matrix(df_us, used_cols)
    mat_au = normalize_rows(mat_au_raw)
    mat_us = normalize_rows(mat_us_raw)

    au_names = df_au["name"].apply(normalize_name).tolist()
    us_names = df_us["name"].apply(normalize_name).tolist()

    # Simple token index for narrowing candidates by overlapping tokens
    token_index = {}
    for idx, name in enumerate(us_names):
        for token in name.split():
            token_index.setdefault(token, set()).add(idx)

    rows = []
    for au_idx, au_row in df_au.iterrows():
        au_name = au_names[au_idx]

        candidate_idxs = set()

        # Nutrient-based top K
        if used_cols:
            nutrient_scores = mat_us @ mat_au[au_idx].reshape(-1, 1)
            nutrient_scores = nutrient_scores.flatten()
            top_nutrient = np.argpartition(-nutrient_scores, kth=min(nutrient_top_k, len(nutrient_scores) - 1))[:nutrient_top_k]
            candidate_idxs.update(top_nutrient.tolist())
        else:
            nutrient_scores = np.zeros(len(df_us))

        # Token overlap candidates
        for token in au_name.split():
            candidate_idxs.update(list(token_index.get(token, [])))
        if token_top_k and len(candidate_idxs) > token_top_k:
            # Keep most promising by nutrient score if available, else arbitrary truncation
            candidate_idxs = set(sorted(candidate_idxs, key=lambda i: nutrient_scores[i] if used_cols else 0, reverse=True)[:token_top_k])

        if not candidate_idxs:
            candidate_idxs = set(range(len(df_us)))  # fallback to all, unlikely but safe

        best_cand = None
        best_score = -1.0
        best_name_score = 0.0
        best_nutrient_score = 0.0

        for us_idx in candidate_idxs:
            n_score = name_similarity(au_name, us_names[us_idx])
            nut_score = nutrient_scores[us_idx] if used_cols else 0.0
            combined = weight_name * n_score + (1 - weight_name) * nut_score
            if combined > best_score:
                best_score = combined
                best_name_score = n_score
                best_nutrient_score = nut_score
                best_cand = us_idx

        rows.append(
            {
                "au_id": au_row["id"],
                "au_name": au_row["name"],
                "us_id": df_us.loc[best_cand, "id"] if best_cand is not None else None,
                "us_name": df_us.loc[best_cand, "name"] if best_cand is not None else None,
                "combined_score": best_score if best_score >= 0 else np.nan,
                "name_score": best_name_score,
                "nutrient_score": best_nutrient_score if used_cols else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Match AUSNUT foods to USDA foods by name and shared nutrients.")
    parser.add_argument("--au", type=Path, default=AU_PATH, help="Path to AUSNUT CSV")
    parser.add_argument("--us", type=Path, default=US_PATH, help="Path to USDA CSV")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Output CSV for matches")
    parser.add_argument("--weight-name", type=float, default=0.6, help="Weight for name similarity vs nutrient similarity")
    args = parser.parse_args()

    df_au = pd.read_csv(args.au)
    df_us = pd.read_csv(args.us)
    if "name" not in df_au or "name" not in df_us:
        raise ValueError("Both AUSNUT and USDA files must have a 'name' column.")
    if "id" not in df_au or "id" not in df_us:
        raise ValueError("Both AUSNUT and USDA files must have an 'id' column.")

    matches = build_matches(df_au, df_us, weight_name=args.weight_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.out, index=False)
    print(f"Wrote {len(matches)} matches to {args.out} (weight_name={args.weight_name})")


if __name__ == "__main__":
    main()
