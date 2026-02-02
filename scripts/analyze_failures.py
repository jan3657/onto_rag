#!/usr/bin/env python3
"""
Analyze failed matches from CRAFT ChEBI evaluation results.

Categorizes failures by:
- Confidence level (high vs low)
- Retrieval stage (1st try, 2nd try, 3rd try)
- Error type (no prediction, wrong prediction, error)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

# Configuration
RESULTS_FILE = Path("data/chebi/results_Qwen-Qwen2.5-14B-Instruct_full_dataset.json")
OUTPUT_DIR = Path("data/chebi/analysis")
HIGH_CONFIDENCE_THRESHOLD = 0.6  # Matches config.CONFIDENCE_THRESHOLD

def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def analyze_failures(data: dict) -> dict:
    """Comprehensive failure analysis."""
    results = data.get('results', [])
    metrics = data.get('metrics', {})
    
    # Separate correct and incorrect
    correct = [r for r in results if r.get('is_correct', False)]
    failures = [r for r in results if not r.get('is_correct', False)]
    
    # === FAILURE CATEGORIES ===
    no_prediction = [f for f in failures if not f.get('predicted_id')]
    wrong_prediction = [f for f in failures if f.get('predicted_id')]
    has_error = [f for f in failures if f.get('error')]
    
    # === CONFIDENCE ANALYSIS ===
    high_conf_failures = []
    low_conf_failures = []
    no_conf_failures = []
    
    for f in failures:
        conf = f.get('confidence')
        if conf is None:
            no_conf_failures.append(f)
        elif conf >= HIGH_CONFIDENCE_THRESHOLD:
            high_conf_failures.append(f)
        else:
            low_conf_failures.append(f)
    
    # Also analyze correct predictions by confidence
    high_conf_correct = [c for c in correct if (c.get('confidence') or 0) >= HIGH_CONFIDENCE_THRESHOLD]
    low_conf_correct = [c for c in correct if (c.get('confidence') or 0) < HIGH_CONFIDENCE_THRESHOLD and c.get('confidence') is not None]
    
    # === QUERY PATTERN ANALYSIS ===
    # Common failed queries (likely generic terms)
    failed_queries = Counter(f.get('query', '').lower() for f in failures)
    top_failed_queries = failed_queries.most_common(30)
    
    # === GOLD ID ANALYSIS ===
    # Which gold IDs are most commonly missed?
    missed_gold_ids = Counter()
    for f in failures:
        for gid in f.get('gold_ids', []):
            missed_gold_ids[gid] += 1
    top_missed_gold = missed_gold_ids.most_common(20)
    
    # === CANDIDATE ANALYSIS ===
    # How many failures had candidates vs no candidates?
    failures_with_candidates = [f for f in failures if f.get('candidate_labels')]
    failures_no_candidates = [f for f in failures if not f.get('candidate_labels')]
    
    # === BUILD ANALYSIS REPORT ===
    analysis = {
        "summary": {
            "total_evaluated": metrics.get('total', len(results)),
            "total_correct": metrics.get('correct', len(correct)),
            "total_failures": len(failures),
            "accuracy": metrics.get('accuracy', len(correct) / len(results) if results else 0),
        },
        "failure_breakdown": {
            "no_prediction": len(no_prediction),
            "wrong_prediction": len(wrong_prediction),
            "had_errors": len(has_error),
        },
        "confidence_analysis": {
            "high_confidence_failures": len(high_conf_failures),
            "low_confidence_failures": len(low_conf_failures),
            "no_confidence_failures": len(no_conf_failures),
            "high_confidence_correct": len(high_conf_correct),
            "low_confidence_correct": len(low_conf_correct),
            "high_conf_precision": len(high_conf_correct) / (len(high_conf_correct) + len(high_conf_failures)) 
                if (len(high_conf_correct) + len(high_conf_failures)) > 0 else 0,
        },
        "retrieval_analysis": {
            "failures_with_candidates": len(failures_with_candidates),
            "failures_without_candidates": len(failures_no_candidates),
            "pct_retrieval_issue": len(failures_no_candidates) / len(failures) if failures else 0,
        },
        "top_failed_queries": top_failed_queries,
        "top_missed_gold_ids": top_missed_gold,
    }
    
    return analysis, failures

def categorize_failure_types(failures: list) -> dict:
    """Categorize failures into semantic groups."""
    categories = defaultdict(list)
    
    for f in failures:
        query = f.get('query', '').lower()
        
        # Generic/vague terms
        if query in ['molecule', 'molecules', 'compound', 'compounds', 'chemical', 'chemicals',
                     'substance', 'substances', 'drug', 'drugs', 'salt', 'salts', 'acid', 'acids',
                     'ion', 'ions', 'agent', 'agents', 'factor', 'factors']:
            categories['generic_terms'].append(f)
        # Adjective forms (cholinergic, dopaminergic, etc.)
        elif query.endswith('ic') or query.endswith('ous') or query.endswith('al'):
            categories['adjective_forms'].append(f)
        # Plural forms
        elif query.endswith('s') and not query.endswith('ss'):
            categories['plural_forms'].append(f)
        # Abbreviations (all caps, short)
        elif query.isupper() and len(query) <= 6:
            categories['abbreviations'].append(f)
        else:
            categories['other'].append(f)
    
    return dict(categories)

def export_failures(failures: list, output_dir: Path):
    """Export failures to JSON for further analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full failures export
    with open(output_dir / "all_failures.json", 'w') as f:
        json.dump(failures, f, indent=2)
    
    # High confidence failures (most concerning)
    high_conf = [f for f in failures if (f.get('confidence') or 0) >= HIGH_CONFIDENCE_THRESHOLD]
    with open(output_dir / "high_confidence_failures.json", 'w') as f:
        json.dump(high_conf, f, indent=2)
    
    # No prediction failures
    no_pred = [f for f in failures if not f.get('predicted_id')]
    with open(output_dir / "no_prediction_failures.json", 'w') as f:
        json.dump(no_pred, f, indent=2)
    
    print(f"Exported {len(failures)} failures to {output_dir}")
    print(f"  - all_failures.json")
    print(f"  - high_confidence_failures.json ({len(high_conf)} items)")
    print(f"  - no_prediction_failures.json ({len(no_pred)} items)")

def print_analysis(analysis: dict, categories: dict):
    """Print formatted analysis report."""
    print("\n" + "="*70)
    print("CRAFT ChEBI FAILURE ANALYSIS REPORT")
    print("="*70)
    
    # Summary
    s = analysis['summary']
    print(f"\n📊 SUMMARY")
    print(f"   Total evaluated: {s['total_evaluated']}")
    print(f"   Correct: {s['total_correct']} ({s['accuracy']:.2%})")
    print(f"   Failures: {s['total_failures']} ({1-s['accuracy']:.2%})")
    
    # Failure breakdown
    fb = analysis['failure_breakdown']
    print(f"\n❌ FAILURE BREAKDOWN")
    print(f"   No prediction:    {fb['no_prediction']:>5} ({fb['no_prediction']/s['total_failures']:.1%})")
    print(f"   Wrong prediction: {fb['wrong_prediction']:>5} ({fb['wrong_prediction']/s['total_failures']:.1%})")
    print(f"   Had errors:       {fb['had_errors']:>5} ({fb['had_errors']/s['total_failures']:.1%})")
    
    # Confidence analysis
    ca = analysis['confidence_analysis']
    print(f"\n🎯 CONFIDENCE ANALYSIS (threshold: {HIGH_CONFIDENCE_THRESHOLD})")
    print(f"   High confidence failures: {ca['high_confidence_failures']:>5}")
    print(f"   Low confidence failures:  {ca['low_confidence_failures']:>5}")
    print(f"   No confidence (errors):   {ca['no_confidence_failures']:>5}")
    print(f"   High conf precision:      {ca['high_conf_precision']:.2%}")
    
    # Retrieval analysis
    ra = analysis['retrieval_analysis']
    print(f"\n🔍 RETRIEVAL ANALYSIS")
    print(f"   Failures WITH candidates: {ra['failures_with_candidates']:>5} (LLM selection issue)")
    print(f"   Failures NO candidates:   {ra['failures_without_candidates']:>5} (Retrieval issue)")
    print(f"   % Retrieval issue:        {ra['pct_retrieval_issue']:.1%}")
    
    # Categories
    print(f"\n📂 FAILURE CATEGORIES")
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"   {cat}: {len(items)}")
    
    # Top failed queries
    print(f"\n🔤 TOP FAILED QUERIES")
    for query, count in analysis['top_failed_queries'][:15]:
        print(f"   '{query}': {count} times")
    
    # Top missed gold IDs
    print(f"\n🏷️ TOP MISSED GOLD IDs")
    for gid, count in analysis['top_missed_gold_ids'][:10]:
        print(f"   {gid}: {count} times")
    
    print("\n" + "="*70)

def main():
    print(f"Loading results from: {RESULTS_FILE}")
    data = load_results(RESULTS_FILE)
    
    # Run analysis
    analysis, failures = analyze_failures(data)
    categories = categorize_failure_types(failures)
    
    # Print report
    print_analysis(analysis, categories)
    
    # Export failures
    export_failures(failures, OUTPUT_DIR)
    
    # Save analysis summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "analysis_summary.json", 'w') as f:
        # Convert Counter objects to dicts for JSON serialization
        serializable = {
            **analysis,
            "categories_counts": {k: len(v) for k, v in categories.items()},
        }
        json.dump(serializable, f, indent=2)
    print(f"\nAnalysis summary saved to: {OUTPUT_DIR / 'analysis_summary.json'}")

if __name__ == "__main__":
    main()
