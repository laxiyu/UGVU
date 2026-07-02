"""Experiment 1: Main Results — UGVU vs. Baselines.

Compares:
    - Vision Banana (baseline)
    - Vanilla Prompt (single generation)
    - Majority Vote
    - UGVU (full pipeline: PCU + UGCF)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ugvu.pipeline import UGVUPipeline


def run_exp1(config_path: str = "configs/default.yaml"):
    """Run Exp1: Main results comparison.

    This experiment evaluates the full UGVU pipeline (PCU + UGCF)
    and compares against baselines by varying the consensus method.
    """
    methods = ["majority", "ugcf", "cmcf"]
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Exp1: Method = {method}")
        print(f"{'='*60}")

        pipeline = UGVUPipeline(
            config_path,
            task="semantic_segmentation",
            output_dir=f"outputs/exp1_{method}",
        )
        # Override consensus method
        pipeline.config.consensus.method = method

        if method == "cmcf":
            pipeline.config.consensus.cross_model_models = ["doubao", "qwen"]

        metrics = pipeline.run()
        results[method] = metrics

        print(f"\n{method}: mIoU = {metrics.get('mIoU', 'N/A')}")

    # Summary
    print("\n" + "=" * 60)
    print("Exp1 Summary: Main Results")
    print("=" * 60)
    for method, metrics in results.items():
        print(f"  {method:12s}: mIoU = {metrics.get('mIoU', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    run_exp1(args.config)
