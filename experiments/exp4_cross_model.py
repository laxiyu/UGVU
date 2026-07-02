"""Experiment 4: Cross-Model Fusion (CMCF).

Compares:
    - Doubao only
    - Qwen only
    - Doubao + Qwen (CMCF)

Validates that cross-model fusion improves over single-model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from ugvu.pipeline import UGVUPipeline


def run_exp4(config_path: str = "configs/default.yaml"):
    """Run Exp4: Cross-Model Fusion."""
    configs_to_test = [
        {"name": "Doubao Only", "models": ["doubao"], "method": "ugcf"},
        {"name": "Qwen Only", "models": ["qwen"], "method": "ugcf"},
        {"name": "Doubao + Qwen (CMCF)", "models": ["doubao", "qwen"], "method": "cmcf"},
    ]

    results = {}

    for cfg in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Exp4: {cfg['name']}")
        print(f"{'='*60}")

        pipeline = UGVUPipeline(
            config_path,
            task="semantic_segmentation",
            output_dir=f"outputs/exp4_{cfg['name'].replace(' ', '_').lower()}",
        )
        pipeline.config.consensus.method = cfg["method"]
        pipeline.config.consensus.cross_model_models = cfg["models"]

        metrics = pipeline.run()
        results[cfg["name"]] = metrics

        print(f"  mIoU = {metrics.get('mIoU', 'N/A'):.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Exp4 Summary: Cross-Model Fusion")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"  {name:30s}: mIoU = {metrics.get('mIoU', 'N/A'):.4f}")

    with open("outputs/exp4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    run_exp4(args.config)
