#!/usr/bin/env python3
"""
Compare AU→US matches to ground truth and report accuracy plus mismatches.

Examples:
  python scripts/evaluate_au_us_matches.py \\
    --ground-truth USA_AU_data/au_us_ground_truth.csv \\
    --predictions USA_AU_data/processed/au_us_matches_rag.csv \\
    --limit 15 \\
    --out-csv /tmp/au_us_mismatches.csv
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


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
) -> None:
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
    for preds_for_au in overlap:
        gt_us = ground_truth[preds_for_au[0].au_id]
        for k in hits.keys():
            topk = preds_for_au[:k]
            if any(p.us_id_norm == gt_us for p in topk):
                hits[k] += 1

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
    print(f"Hit@1:                {accuracy:.2%}")
    print(f"Hit@2:                {hit2:.2%}")
    print(f"Hit@3:                {hit3:.2%}")
    print(f"Coverage:             {coverage:.2%}")
    print(f"Correct over GT:      {correct_gt:.2%}")

    return mismatches


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
) -> None:
    fieldnames = [
        "au_id",
        "rank",
        "gt_us_id",
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
            writer.writerow(
                {
                    "au_id": p.au_id,
                    "rank": p.rank,
                    "gt_us_id": ground_truth[p.au_id],
                    "pred_us_id_raw": p.us_id_raw,
                    "pred_us_id_norm": p.us_id_norm,
                    "au_name": p.au_name,
                    "us_name": p.us_name,
                    "confidence": p.confidence,
                }
            )
    print(f"\nMismatch CSV written to: {path}")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    mismatches = evaluate(ground_truth, predictions)
    print_mismatches(mismatches, ground_truth, limit=args.limit)
    if args.out_csv:
        write_mismatches_csv(mismatches, ground_truth, path=args.out_csv)


if __name__ == "__main__":
    main()
