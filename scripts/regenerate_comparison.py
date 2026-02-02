#!/usr/bin/env python3
"""
Regenerate comparison.json from existing benchmark result directories.

This script scans all directories in data/benchmark_results/ and rebuilds
the comparison.json file based on their summary.json files.

Usage:
    python scripts/regenerate_comparison.py
"""

import json
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark_results"
COMPARISON_FILE = RESULTS_DIR / "comparison.json"


def regenerate_comparison():
    """Scan result directories and rebuild comparison.json."""
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return
    
    # Initialize comparison structure
    comparison = {
        "models": {},
        "last_updated": datetime.now().isoformat()
    }
    
    # Find all result directories (exclude comparison.json itself)
    result_dirs = [d for d in RESULTS_DIR.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(result_dirs)} result directories")
    print("=" * 60)
    
    # Process each directory
    for result_dir in sorted(result_dirs):
        summary_file = result_dir / "summary.json"
        
        if not summary_file.exists():
            print(f"⚠️  Skipping {result_dir.name}: no summary.json found")
            continue
        
        try:
            with summary_file.open("r") as f:
                summary = json.load(f)
            
            # Extract metadata
            run_key = result_dir.name
            model = summary.get("model", "unknown")
            provider = summary.get("provider", "unknown")
            timestamp = summary.get("timestamp", "unknown")
            sample_size = summary.get("sample_size")
            datasets = summary.get("datasets", {})
            
            # Add to comparison
            comparison["models"][run_key] = {
                "provider": provider,
                "model": model,
                "timestamp": timestamp,
                "run_dir": result_dir.name,
                "sample_size": sample_size,
                "datasets": datasets,
            }
            
            # Print summary
            print(f"✓ {run_key}")
            print(f"  Model: {model}")
            print(f"  Provider: {provider}")
            print(f"  Timestamp: {timestamp}")
            
            # Show accuracy for each dataset
            for ds_key, metrics in datasets.items():
                if "error" in metrics:
                    print(f"    {ds_key}: ERROR")
                elif "accuracy" in metrics:
                    acc = metrics["accuracy"]
                    total = metrics.get("total", 0)
                    correct = metrics.get("correct", 0)
                    print(f"    {ds_key}: {correct}/{total} = {acc:.2%}")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Error processing {result_dir.name}: {e}")
    
    # Save comparison file
    print("=" * 60)
    
    if comparison["models"]:
        with COMPARISON_FILE.open("w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Successfully regenerated: {COMPARISON_FILE}")
        print(f"  Total models: {len(comparison['models'])}")
    else:
        print("⚠️  No valid result directories found. comparison.json not created.")


if __name__ == "__main__":
    regenerate_comparison()
