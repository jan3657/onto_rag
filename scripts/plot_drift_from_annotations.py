#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate(records: List[Dict[str, Any]]):
    """
    Extract class/direction info from records.
    Returns:
      - class_counts: Counter of annotation.class
      - class_dir_counts: nested dict[class][direction] = count (direction may be null)
      - directions: sorted list of non-null directions
    """
    class_counts = Counter()
    class_dir_counts = defaultdict(Counter)
    directions_set = set()

    for r in records:
        ann = r.get("annotation") or {}
        cls = ann.get("class")
        direction = ann.get("direction")

        if not cls:
            continue

        class_counts[cls] += 1
        if direction is not None:
            class_dir_counts[cls][direction] += 1
            directions_set.add(direction)

    directions = sorted(directions_set)
    return class_counts, class_dir_counts, directions


def plot_class_distribution(class_counts: Counter, out_path: str):
    """
    Simple bar chart: count per class (including EXACT_MATCH).
    """
    if not class_counts:
        return

    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    x = range(len(classes))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")

    ax.set_ylabel("Count")
    ax.set_title("Distribution of annotation classes")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_stacked_by_class(class_dir_counts: Dict[str, Counter],
                          directions: List[str],
                          out_path: str,
                          exclude_exact: bool = True):
    """
    100% stacked bar: for each class, proportion of directions.
    Typically:
      - rows = non-EXACT_MATCH classes
      - columns = directions (Taxon→Product, Product→Taxon, etc.)
    """
    if not directions:
        return

    # Decide which classes to include
    classes = sorted(class_dir_counts.keys())
    if exclude_exact:
        classes = [c for c in classes if c != "EXACT_MATCH"]

    if not classes:
        return

    # Build DataFrame of proportions per class
    data = {d: [] for d in directions}
    index = []

    for cls in classes:
        dir_counts = class_dir_counts.get(cls, {})
        total = sum(dir_counts.values())
        if total == 0:
            # No directional info for this class; skip
            continue
        index.append(cls)
        for d in directions:
            data[d].append(dir_counts.get(d, 0) / total)

    if not index:
        return

    df = pd.DataFrame(data, index=index)

    fig, ax = plt.subplots(figsize=(9, 4))
    left = [0.0] * len(df.index)
    x = range(len(df.index))

    for d in directions:
        vals = df[d].values
        ax.bar(x, vals, bottom=left, label=d)
        left = [left[i] + vals[i] for i in range(len(vals))]

    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion within class")
    ax.set_title("Ontology drift directions by class (non-EXACT cases)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(len(directions), 4), frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df


def save_csvs(class_counts: Counter,
              class_dir_counts: Dict[str, Counter],
              directions: List[str],
              out_dir: str):
    """
    Save underlying numeric tables for reproducibility.
    """
    # Class counts
    class_counts_path = os.path.join(out_dir, "class_counts.csv")
    with open(class_counts_path, "w", encoding="utf-8") as f:
        f.write("class,count\n")
        for cls, cnt in class_counts.items():
            f.write(f"{cls},{cnt}\n")

    # Class-direction counts
    class_dir_counts_path = os.path.join(out_dir, "class_direction_counts.csv")
    all_classes = sorted(class_dir_counts.keys())
    with open(class_dir_counts_path, "w", encoding="utf-8") as f:
        f.write("class,direction,count\n")
        for cls in all_classes:
            for d in directions:
                cnt = class_dir_counts[cls].get(d, 0)
                if cnt > 0:
                    f.write(f"{cls},{d},{cnt}\n")

    # Class-direction proportions (100% per class, only where total>0)
    class_dir_props_path = os.path.join(out_dir, "class_direction_proportions.csv")
    with open(class_dir_props_path, "w", encoding="utf-8") as f:
        f.write("class,direction,proportion\n")
        for cls in all_classes:
            total = sum(class_dir_counts[cls].values())
            if total == 0:
                continue
            for d in directions:
                cnt = class_dir_counts[cls].get(d, 0)
                if cnt > 0:
                    prop = cnt / total
                    f.write(f"{cls},{d},{prop:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ontology drift annotations from JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cafeteria_results/formated_results/cafeteria_misses_drift_annotations_gemini.jsonl",
        help="Path to JSONL file with annotations."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="cafeteria_results/formated_results/plots",
        help="Directory to store plots and CSVs."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    records = load_jsonl(args.input)
    if not records:
        raise SystemExit(f"No records loaded from {args.input}")

    class_counts, class_dir_counts, directions = aggregate(records)

    # Save CSVs
    save_csvs(class_counts, class_dir_counts, directions, args.outdir)

    # Plot 1: distribution of classes
    plot_class_distribution(
        class_counts,
        os.path.join(args.outdir, "class_distribution.png")
    )

    # Plot 2: stacked proportions by class (non-EXACT)
    df_props = plot_stacked_by_class(
        class_dir_counts,
        directions,
        os.path.join(args.outdir, "drift_stacked_by_class.png"),
        exclude_exact=True,
    )

    # Optional: also save the stacked-bar proportions table if created via df
    if df_props is not None:
        df_props.to_csv(
            os.path.join(args.outdir, "drift_stacked_by_class_proportions.csv"),
            index_label="class"
        )


if __name__ == "__main__":
    main()
