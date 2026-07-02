"""Experiment 3: Generation Consistency.

Tests how performance varies with K (number of samples).

Compares K ∈ {1, 3, 5, 10} for both Majority Vote and UGCF.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json

from ugvu.pipeline import UGVUPipeline


def run_exp3(config_path: str = "configs/default.yaml"):
    """Run Exp3: Generation consistency across K values."""
    k_values = [1, 3, 5, 10]
    methods = ["majority", "ugcf"]
    results = {}

    for method in methods:
        results[method] = {}
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Exp3: Method={method}, K={k}")
            print(f"{'='*60}")

            pipeline = UGVUPipeline(
                config_path,
                task="semantic_segmentation",
                output_dir=f"outputs/exp3_{method}_k{k}",
            )
            pipeline.config.sampling.k_samples = k
            pipeline.config.consensus.method = method

            metrics = pipeline.run()
            results[method][k] = metrics

            print(f"  mIoU = {metrics.get('mIoU', 'N/A'):.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("Exp3 Summary: Generation Consistency")
    print("=" * 70)
    print(f"{'K':<6} {'Majority mIoU':<16} {'UGCF mIoU':<16}")
    print("-" * 38)
    for k in k_values:
        maj = results["majority"][k].get("mIoU", 0)
        ugcf = results["ugcf"][k].get("mIoU", 0)
        print(f"{k:<6} {maj:<16.4f} {ugcf:<16.4f}")

    # Save
    serializable = {}
    for method in methods:
        serializable[method] = {str(k): v for k, v in results[method].items()}
    with open("outputs/exp3_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    run_exp3(args.config)
