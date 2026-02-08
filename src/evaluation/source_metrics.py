"""
Utility functions for calculating source attribution metrics.

Helps analyze which retrieval method (lexical, MiniLM, SapBERT) contributed
to correct vs incorrect predictions.
"""

from typing import List, Dict, Any


def calculate_source_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics showing which retrieval source(s) contributed predictions.

    Args:
        results: List of result dictionaries, each containing:
            - predicted_sources: List of source names (e.g., ["lexical", "sapbert"])
            - is_correct: Boolean indicating if prediction was correct

    Returns:
        Dictionary with source attribution statistics:
        - by_source: Counts for each source combination
        - correct_percentages: Percentage of correct predictions from each source
        - incorrect_percentages: Percentage of incorrect predictions from each source
        - precision_by_source: Precision for each source combination
    """
    source_stats = {
        "lexical_only": {"correct": 0, "incorrect": 0},
        "minilm_only": {"correct": 0, "incorrect": 0},
        "sapbert_only": {"correct": 0, "incorrect": 0},
        "lexical+minilm": {"correct": 0, "incorrect": 0},
        "lexical+sapbert": {"correct": 0, "incorrect": 0},
        "minilm+sapbert": {"correct": 0, "incorrect": 0},
        "all_three": {"correct": 0, "incorrect": 0},
        "unknown": {"correct": 0, "incorrect": 0},
    }

    for r in results:
        sources = r.get("predicted_sources")

        # Determine source combination key
        if not sources:
            key = "unknown"
        else:
            # Sort for consistent key naming
            sources_sorted = sorted(sources)
            if sources_sorted == ["lexical"]:
                key = "lexical_only"
            elif sources_sorted == ["minilm"]:
                key = "minilm_only"
            elif sources_sorted == ["sapbert"]:
                key = "sapbert_only"
            elif sources_sorted == ["lexical", "minilm"]:
                key = "lexical+minilm"
            elif sources_sorted == ["lexical", "sapbert"]:
                key = "lexical+sapbert"
            elif sources_sorted == ["minilm", "sapbert"]:
                key = "minilm+sapbert"
            elif len(sources_sorted) == 3:
                key = "all_three"
            else:
                key = "unknown"

        # Increment counters
        if r.get("is_correct"):
            source_stats[key]["correct"] += 1
        else:
            source_stats[key]["incorrect"] += 1

    # Calculate totals
    total_correct = sum(s["correct"] for s in source_stats.values())
    total_incorrect = sum(s["incorrect"] for s in source_stats.values())

    # Calculate percentages and precision
    correct_percentages = {}
    incorrect_percentages = {}
    precision_by_source = {}

    for key, stats in source_stats.items():
        correct_count = stats["correct"]
        incorrect_count = stats["incorrect"]
        total_for_source = correct_count + incorrect_count

        # Percentage of all correct/incorrect predictions
        correct_percentages[key] = round(correct_count / total_correct * 100, 2) if total_correct > 0 else 0
        incorrect_percentages[key] = round(incorrect_count / total_incorrect * 100, 2) if total_incorrect > 0 else 0

        # Precision for this source combination
        precision_by_source[key] = round(correct_count / total_for_source * 100, 2) if total_for_source > 0 else 0

    return {
        "by_source": source_stats,
        "correct_percentages": correct_percentages,
        "incorrect_percentages": incorrect_percentages,
        "precision_by_source": precision_by_source,
        "summary": {
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
        }
    }


def calculate_retrieval_source_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate retrieval success by source across pipeline attempts.

    Expects each result to optionally include:
      - gold_found_by_source_any_attempt: {
            "lexical": bool, "minilm": bool, "sapbert": bool, "merged_top10": bool
        }
      - gold_covered_by_index: bool
    """
    total = len(results)
    covered = [r for r in results if r.get("gold_covered_by_index") is True]
    covered_total = len(covered)

    sources = ["lexical", "minilm", "sapbert", "merged_top10"]
    by_source: Dict[str, Dict[str, Any]] = {}

    for source in sources:
        hits_all = sum(
            1
            for r in results
            if (r.get("gold_found_by_source_any_attempt") or {}).get(source)
        )
        hits_covered = sum(
            1
            for r in covered
            if (r.get("gold_found_by_source_any_attempt") or {}).get(source)
        )
        by_source[source] = {
            "hits_all": hits_all,
            "rate_all": round(hits_all / total, 4) if total > 0 else 0.0,
            "hits_covered": hits_covered,
            "rate_covered": round(hits_covered / covered_total, 4) if covered_total > 0 else 0.0,
        }

    vector_hits_all = sum(
        1
        for r in results
        if ((r.get("gold_found_by_source_any_attempt") or {}).get("minilm")
            or (r.get("gold_found_by_source_any_attempt") or {}).get("sapbert"))
    )
    vector_hits_covered = sum(
        1
        for r in covered
        if ((r.get("gold_found_by_source_any_attempt") or {}).get("minilm")
            or (r.get("gold_found_by_source_any_attempt") or {}).get("sapbert"))
    )

    return {
        "by_source": by_source,
        "vector_union": {
            "hits_all": vector_hits_all,
            "rate_all": round(vector_hits_all / total, 4) if total > 0 else 0.0,
            "hits_covered": vector_hits_covered,
            "rate_covered": round(vector_hits_covered / covered_total, 4) if covered_total > 0 else 0.0,
        },
        "summary": {
            "total_items": total,
            "covered_items": covered_total,
            "uncovered_items": total - covered_total,
        },
    }
