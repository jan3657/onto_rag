#!/usr/bin/env python3
"""
Build a unified per-query dataframe from all benchmark runs.

Outputs:
- data/analysis/errors.parquet  (full detail, compressed)
- data/analysis/errors.csv      (lightweight, no candidate_labels)

Columns (core):
  run_id, model, provider, timestamp, dataset
  query, gold_ids, predicted_id, predicted_label, is_correct, confidence
  candidate_count, gold_in_candidates, gold_first_found_at_attempt, total_retrieval_attempts
  from_cache, error
  error_type (derived), lexical_* helpers
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path("data/benchmark_results")
OUT_DIR = Path("data/analysis")
OUT_FULL = OUT_DIR / "errors.parquet"
OUT_CSV = OUT_DIR / "errors.csv"


@dataclass
class RunMeta:
    run_id: str
    model: Optional[str]
    provider: Optional[str]
    timestamp: Optional[str]
    sample_size: Optional[int]
    concurrent_requests: Optional[int]


def load_summary(run_dir: Path) -> RunMeta:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return RunMeta(run_dir.name, None, None, None, None, None)
    with open(summary_path) as f:
        data = json.load(f)
    # concurrent_requests lives under each dataset; take max to be safe
    conc = None
    try:
        conc = max(
            v.get("concurrent_requests")
            for v in data.get("datasets", {}).values()
            if isinstance(v, dict) and "concurrent_requests" in v
        )
    except Exception:
        conc = None
    return RunMeta(
        run_id=run_dir.name,
        model=data.get("model"),
        provider=data.get("provider"),
        timestamp=data.get("timestamp"),
        sample_size=data.get("sample_size"),
        concurrent_requests=conc,
    )


def iter_results(run_dir: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield result rows from results.jsonl; if missing, fall back to details.json -> ["results"].
    """
    jsonl_path = run_dir / "results.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)
        return

    # Fallback: details.json contains a dict with dataset keys and "results" lists
    details_path = run_dir / "details.json"
    if details_path.exists():
        with open(details_path) as f:
            details = json.load(f)
        for dataset_block in details.values():
            if not isinstance(dataset_block, dict):
                continue
            for row in dataset_block.get("results", []):
                yield row
        return

    raise FileNotFoundError(f"No results.jsonl or details.json in {run_dir}")


def derive_error_type(row: Dict[str, Any]) -> str:
    if row.get("is_correct"):
        return "correct"
    if row.get("error"):
        return "system_error"
    if row.get("predicted_id") is None:
        return "no_prediction"
    if row.get("gold_in_candidates") is False:
        return "retrieval_miss"
    gold_ids = row.get("gold_ids") or []
    if gold_ids and row.get("predicted_id") not in gold_ids:
        return "ranking_miss"
    return "other"


def add_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    greek_re = re.compile(r"(alpha|beta|gamma|delta|epsilon|θ|λ|μ|ν|π|σ|φ|χ|ψ|ω|α|β|γ)", re.IGNORECASE)

    def has_greek(text: str) -> bool:
        return bool(greek_re.search(text))

    df["query_lower"] = df["query"].str.lower()
    df["query_len"] = df["query"].str.len()
    df["query_tokens"] = df["query"].str.split().str.len()
    df["query_has_digit"] = df["query"].str.contains(r"\d")
    df["query_has_hyphen"] = df["query"].str.contains("-")
    df["query_is_upper"] = df["query"].apply(lambda x: x.isupper() and len(x) > 1)
    df["query_has_greek"] = df["query"].apply(has_greek)
    return df


def build_dataframe(root: Path = ROOT) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = load_summary(run_dir)
        try:
            for r in iter_results(run_dir):
                r = dict(r)  # copy
                r.update(
                    {
                        "run_id": meta.run_id,
                        "model": meta.model,
                        "provider": meta.provider,
                        "timestamp": meta.timestamp,
                        "sample_size": meta.sample_size,
                        "concurrent_requests": meta.concurrent_requests,
                    }
                )
                r["error_type"] = derive_error_type(r)
                rows.append(r)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue
    if not rows:
        raise SystemExit("No rows collected; aborting.")

    df = pd.DataFrame(rows)
    df = add_lexical_features(df)
    return df


def write_outputs(df: pd.DataFrame, out_parquet: Path = OUT_FULL, out_csv: Path = OUT_CSV) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    # Light CSV (drop heavy columns)
    light = df.drop(columns=["candidate_labels"], errors="ignore")
    light.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows -> {out_parquet}")
    print(f"Wrote {len(light):,} rows -> {out_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build unified error dataframe.")
    ap.add_argument("--root", type=Path, default=ROOT, help="Path to benchmark_results directory")
    ap.add_argument("--out-parquet", type=Path, default=OUT_FULL, help="Full parquet output path")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV, help="Light CSV output path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataframe(args.root)
    write_outputs(df, args.out_parquet, args.out_csv)


if __name__ == "__main__":
    main()
