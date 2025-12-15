#!/usr/bin/env python3
"""
Compare AU→US matches to ground truth and report accuracy plus mismatches.

Examples:
  python scripts/evaluate_au_us_matches.py \\
    --ground-truth USA_AU_data/au_us_ground_truth.csv \\
    --predictions USA_AU_data/processed/au_us_matches_rag.csv \\
    --limit 15 \\
    --out-csv /tmp/au_us_mismatches.csv \\
    --top3-miss-csv USA_AU_data/processed/missmatches.csv
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Prediction:
    au_id: str
    us_id_raw: str
    us_id_norm: str
    au_name: str
    us_name: str
    confidence: Optional[float]
    rank: Optional[int] = None


def normalize_us_id(us_id_raw: str) -> str:
    """Strip USDA prefix like 'USDA:170924'."""
    cleaned = us_id_raw.strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[1]
    return cleaned


def load_ground_truth(path: str) -> Dict[str, str]:
    au_to_us: Dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            au_id = row["au_id"].strip()
            us_id = row["us_id"].strip()
            if au_id in au_to_us and au_to_us[au_id] != us_id:
                # Keep first but warn inline for visibility.
                print(
                    f"Warning: duplicate ground truth for {au_id}: "
                    f"{au_to_us[au_id]} vs {us_id}"
                )
            au_to_us.setdefault(au_id, us_id)
    return au_to_us


def load_predictions(path: str) -> List[Prediction]:
    preds: List[Prediction] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            au_id = row["au_id"].strip()
            us_id_raw = row["us_id"].strip()
            rank_raw = row.get("rank")
            try:
                rank = int(rank_raw) if rank_raw not in (None, "",) else None
            except (TypeError, ValueError):
                rank = None
            preds.append(
                Prediction(
                    au_id=au_id,
                    us_id_raw=us_id_raw,
                    us_id_norm=normalize_us_id(us_id_raw),
                    au_name=row.get("au_name", "").strip(),
                    us_name=row.get("us_name", "").strip(),
                    confidence=_safe_float(row.get("confidence_score")),
                    rank=rank,
                )
            )
    return preds


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(
    ground_truth: Dict[str, str], predictions: Iterable[Prediction]
) -> Tuple[List[Prediction], List[List[Prediction]]]:
    preds = list(predictions)
    gt_count = len(ground_truth)
    pred_count = len(preds)

    # group predictions per au_id and sort by rank (or file order)
    grouped: Dict[str, List[Prediction]] = {}
    for idx, p in enumerate(preds):
        grouped.setdefault(p.au_id, []).append((p, idx))
    for au, lst in grouped.items():
        grouped[au] = [
            t[0]
            for t in sorted(
                lst,
                key=lambda t: (t[0].rank if t[0].rank is not None else 9999, t[1]),
            )
        ]

    overlap_ids = [au for au in grouped.keys() if au in ground_truth]
    overlap = [grouped[au] for au in overlap_ids]

    hits = {1: 0, 2: 0, 3: 0}
    top3_misses: List[List[Prediction]] = []
    for preds_for_au in overlap:
        gt_us = ground_truth[preds_for_au[0].au_id]
        for k in hits.keys():
            topk = preds_for_au[:k]
            if any(p.us_id_norm == gt_us for p in topk):
                hits[k] += 1
        top3 = preds_for_au[:3]
        if not any(p.us_id_norm == gt_us for p in top3):
            top3_misses.append(top3)

    denom = len(overlap) if overlap else 0
    accuracy = hits[1] / denom if denom else 0.0
    hit2 = hits[2] / denom if denom else 0.0
    hit3 = hits[3] / denom if denom else 0.0

    coverage = len(overlap) / gt_count if gt_count else 0.0
    correct_gt = hits[1] / gt_count if gt_count else 0.0

    # mismatches based on top-1 choice
    mismatches = []
    for preds_for_au in overlap:
        top1 = preds_for_au[0]
        if ground_truth[top1.au_id] != top1.us_id_norm:
            mismatches.append(top1)

    print("=== Summary ===")
    print(f"Ground truth entries: {gt_count}")
    print(f"Predicted entries:    {pred_count}")
    print(f"Overlap (by au_id):   {len(overlap)}")
    print(f"Top-3 misses:         {len(top3_misses)}")
    print(f"Hit@1:                {accuracy:.2%}")
    print(f"Hit@2:                {hit2:.2%}")
    print(f"Hit@3:                {hit3:.2%}")
    print(f"Coverage:             {coverage:.2%}")
    print(f"Correct over GT:      {correct_gt:.2%}")

    return mismatches, top3_misses


def print_mismatches(
    mismatches: List[Prediction],
    ground_truth: Dict[str, str],
    limit: int,
) -> None:
    if not mismatches:
        print("\nNo mismatches found.")
        return

    print(f"\n=== First {min(limit, len(mismatches))} mismatches ===")
    for p in mismatches[:limit]:
        gt_us = ground_truth[p.au_id]
        conf = f"{p.confidence:.2f}" if p.confidence is not None else "n/a"
        print(
            f"au_id {p.au_id} | rank {p.rank or '?'} | gt {gt_us} | pred {p.us_id_raw} "
            f"| au_name '{p.au_name}' | us_name '{p.us_name}' "
            f"| confidence {conf}"
        )


def write_mismatches_csv(
    mismatches: List[Prediction],
    ground_truth: Dict[str, str],
    path: str,
    label: str = "Mismatch",
    us_names: Optional[Dict[str, str]] = None,
) -> None:
    fieldnames = [
        "au_id",
        "rank",
        "gt_us_id",
        "gt_us_name",
        "pred_us_id_raw",
        "pred_us_id_norm",
        "au_name",
        "us_name",
        "confidence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in mismatches:
            gt_us_id = ground_truth[p.au_id]
            writer.writerow(
                {
                    "au_id": p.au_id,
                    "rank": p.rank,
                    "gt_us_id": gt_us_id,
                    "gt_us_name": (us_names or {}).get(gt_us_id, ""),
                    "pred_us_id_raw": p.us_id_raw,
                    "pred_us_id_norm": p.us_id_norm,
                    "au_name": p.au_name,
                    "us_name": p.us_name,
                    "confidence": p.confidence,
                }
            )
    print(f"\n{label} CSV written to: {path}")


def load_us_names(path: Optional[str]) -> Dict[str, str]:
    """Load USDA id → name mapping if available."""
    if not path:
        return {}
    us_names: Dict[str, str] = {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            id_field = "id" if "id" in fieldnames else "us_id" if "us_id" in fieldnames else None
            name_field = "name" if "name" in fieldnames else None
            if not id_field or not name_field:
                return {}
            for row in reader:
                us_id = (row.get(id_field) or "").strip()
                us_name = (row.get(name_field) or "").strip()
                if us_id:
                    us_names[us_id] = us_name
    except FileNotFoundError:
        print(f"Warning: USDA names file not found: {path}")
    return us_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AU-US predicted matches to ground truth and show accuracy."
        )
    )
    parser.add_argument(
        "--ground-truth",
        default="USA_AU_data/au_us_ground_truth.csv",
        help="Path to ground truth CSV (columns: au_id, us_id).",
    )
    parser.add_argument(
        "--predictions",
        default="USA_AU_data/processed/au_us_matches_rag_top3.csv",
        help="Path to predictions CSV (must include au_id, us_id).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many mismatches to print to stdout.",
    )
    parser.add_argument(
        "--out-csv",
        help="Optional path to write all mismatches as CSV.",
    )
    parser.add_argument(
        "--top3-miss-csv",
        default="USA_AU_data/processed/missmatches.csv",
        help=(
            "Path to write rows where the ground truth is missing from the top-3 "
            "predictions (default: USA_AU_data/processed/missmatches.csv). Use an "
            "empty string to skip."
        ),
    )
    parser.add_argument(
        "--usda-names",
        default="USA_AU_data/processed/usda_complete.csv",
        help=(
            "Optional USDA CSV with columns including id and name; used to add "
            "gt_us_name to outputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    us_names = load_us_names(args.usda_names)
    mismatches, top3_misses = evaluate(ground_truth, predictions)
    print_mismatches(mismatches, ground_truth, limit=args.limit)
    if args.out_csv:
        write_mismatches_csv(
            mismatches, ground_truth, path=args.out_csv, us_names=us_names
        )
    if args.top3_miss_csv:
        top3_flat: List[Prediction] = [
            p for group in top3_misses for p in group
        ]
        if top3_flat:
            write_mismatches_csv(
                top3_flat,
                ground_truth,
                path=args.top3_miss_csv,
                label="Top-3 miss",
                us_names=us_names,
            )
        else:
            print("\nNo top-3 misses to write.")


if __name__ == "__main__":
    main()
