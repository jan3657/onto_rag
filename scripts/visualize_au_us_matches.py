"""
Generate human-friendly comparison charts for AU→US matches.

Capabilities:
- Plot AU vs predicted USDA nutrients (existing behavior).
- If given ground truth + USDA nutrition table, also overlay the ground truth USDA
  nutrients and optionally focus only on mismatches.

Reads:
- USA_AU_data/processed/au_us_matches_rag.csv (produced by scripts/map_au_to_us.py)
- USA_AU_data/au_us_ground_truth.csv (optional, to overlay GT)
- USA_AU_data/processed/usda_complete.csv (only needed when GT overlay is used)

Writes:
- PNG bar charts per match under USA_AU_data/processed/au_us_plots/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


MATCHES_CSV = Path("USA_AU_data/processed/au_us_matches_rag.csv")
GROUND_TRUTH_CSV = Path("USA_AU_data/au_us_ground_truth.csv")
USDA_CSV = Path("USA_AU_data/processed/usda_complete.csv")
PLOTS_DIR = Path("USA_AU_data/processed/au_us_plots")

NUTRIENTS = [
    "prot_g",
    "fat_g",
    "fasat_g",
    "sugar_g",
    "fibt_g",
    "na_mg",
    "fe_mg",
    "p_mg",
    "energy_kj",  # normalized energy field from map_au_to_us
]


def to_num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def normalize_us_id(us_id: str) -> str:
    cleaned = (us_id or "").strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[1]
    return cleaned


def load_ground_truth(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    import csv

    au_to_us: Dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            au_id = row["au_id"].strip()
            us_id = row["us_id"].strip()
            au_to_us[au_id] = us_id
    return au_to_us


def load_usda(path: Path) -> Dict[str, Dict[str, float]]:
    """Return dict of USDA id → nutrient values + metadata."""
    if not path.exists():
        return {}
    import csv

    wanted = {
        "energy_kj": "enerakj",
        "prot_g": "prot_g",
        "fat_g": "fat_g",
        "fasat_g": "fasat_g",
        "sugar_g": "sugar_g",
        "fibt_g": "fibt_g",
        "na_mg": "na_mg",
        "fe_mg": "fe_mg",
        "p_mg": "p_mg",
    }
    data: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            us_id = row.get("id")
            if not us_id:
                continue
            record = {
                "name": row.get("name", ""),
                "classification": row.get("classification", ""),
            }
            for out_key, in_key in wanted.items():
                record[out_key] = to_num(row.get(in_key))
            data[us_id.strip()] = record
    return data


def plot_match(
    row: pd.Series,
    nutrients: List[str],
    out_dir: Path,
    gt_us: Optional[Dict[str, float]] = None,
    gt_us_id: Optional[str] = None,
    mismatch: bool = False,
):
    au_name = row.get("au_name", "AU item")
    us_name = row.get("us_name", "USDA item")
    match_id = row.get("au_id", "unknown")
    rank = row.get("rank")
    confidence = row.get("confidence_score", 0)

    au_vals = [to_num(row.get(f"au_{n}")) for n in nutrients]
    pred_vals = [to_num(row.get(f"us_{n}")) for n in nutrients]
    gt_vals = [gt_us.get(n, 0.0) for n in nutrients] if gt_us else None

    series = [("AU", au_vals), ("Retrieved", pred_vals)]
    if gt_vals:
        series.append(("HandAnnotated", gt_vals))

    num_series = len(series)
    width = 0.8 / num_series
    offsets = [
        (idx - (num_series - 1) / 2) * width for idx in range(num_series)
    ]

    x = range(len(nutrients))
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, (label, values) in zip(offsets, series):
        ax.bar([i + offset for i in x], values, width, label=label)

    lines = [
        f"AU: {au_name}",
        f"Retrieved: {us_name}",
    ]
    if gt_us_id:
        gt_name = gt_us.get("name", "") if gt_us else ""
        lines.append(f"HandAnnotated: {gt_name}")
    if rank:
        lines.append(f"Rank: {rank}")
    lines.append(f"Confidence: {confidence:.2f}")
    title = "\n".join(lines)

    ax.set_ylabel("Value (log scale)")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nutrients, rotation=45, ha="right")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    out_dir.mkdir(parents=True, exist_ok=True)
    if gt_us_id:
        suffix = "_mismatch" if mismatch else "_gt"
    else:
        suffix = ""
    rank_part = f"_r{rank}" if rank else ""
    out_path = out_dir / f"{match_id}{rank_part}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(
    limit: Optional[int],
    min_conf: Optional[float],
    ground_truth_path: Optional[Path],
    usda_path: Optional[Path],
    only_mismatches: bool,
):
    df = pd.read_csv(MATCHES_CSV)
    df = df.dropna(subset=["us_id"])
    if min_conf is None:
        min_conf = 0.8
    df = df[df["confidence_score"].fillna(0) >= min_conf]
    if limit:
        df = df.head(limit)

    gt_map: Dict[str, str] = {}
    usda_data: Dict[str, Dict[str, float]] = {}
    if ground_truth_path:
        gt_map = load_ground_truth(ground_truth_path)
    if gt_map and usda_path:
        usda_data = load_usda(usda_path)

    def include_row(r: pd.Series) -> bool:
        if not only_mismatches or not gt_map:
            return True
        gt_us = gt_map.get(r["au_id"])
        pred_us = normalize_us_id(r["us_id"])
        return bool(gt_us) and gt_us != pred_us

    df = df[df.apply(include_row, axis=1)]

    if df.empty:
        print("No matches to plot (check filters).")
        return

    print(f"Plotting {len(df)} matches to {PLOTS_DIR} ...")
    for _, row in df.iterrows():
        gt_us_id = gt_map.get(row["au_id"]) if gt_map else None
        gt_us_row = usda_data.get(gt_us_id) if gt_us_id else None
        pred_us_norm = normalize_us_id(row["us_id"])
        mismatch = bool(gt_us_id and gt_us_id != pred_us_norm)
        out_path = plot_match(
            row,
            NUTRIENTS,
            PLOTS_DIR,
            gt_us=gt_us_row,
            gt_us_id=gt_us_id,
            mismatch=mismatch,
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize AU→US matches with nutrient bar charts."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of matches to plot."
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum confidence to include (default: 0.8).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=GROUND_TRUTH_CSV,
        help="Ground truth CSV (au_id, us_id). Leave empty to skip overlay.",
    )
    parser.add_argument(
        "--usda",
        type=Path,
        default=USDA_CSV,
        help="USDA nutrient CSV (needed for GT overlay).",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="If ground truth provided, only plot rows where prediction != ground truth.",
    )
    args = parser.parse_args()
    main(
        limit=args.limit,
        min_conf=args.min_conf,
        ground_truth_path=args.ground_truth,
        usda_path=args.usda,
        only_mismatches=args.only_mismatches,
    )
