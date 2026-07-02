"""Experiment 6: Ablation Study.

Systematically removes each contribution:
    - w/o PCU (no uncertainty weighting → majority vote)
    - w/o UGCF (use simple average instead of uncertainty weights)
    - w/o CMCF (single model only)

Quantifies the contribution of each component.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from ugvu.pipeline import UGVUPipeline


def run_exp6(config_path: str = "configs/default.yaml"):
    """Run Exp6: Ablation study."""
    ablations = [
        {"name": "Full UGVU (PCU + UGCF + CMCF)", "method": "cmcf", "uncertainty": "entropy",
         "models": ["doubao", "qwen"]},
        {"name": "w/o CMCF (single model UGCF)", "method": "ugcf", "uncertainty": "entropy",
         "models": ["doubao"]},
        {"name": "w/o UGCF (majority vote)", "method": "majority", "uncertainty": None,
         "models": ["doubao"]},
        {"name": "w/o PCU (single generation, baseline)", "method": "majority", "uncertainty": None,
         "models": ["doubao"], "k": 1},
    ]

    results = {}

    for cfg in ablations:
        print(f"\n{'='*60}")
        print(f"Exp6: {cfg['name']}")
        print(f"{'='*60}")

        pipeline = UGVUPipeline(
            config_path,
            task="semantic_segmentation",
            output_dir=f"outputs/exp6_{cfg['name'].replace(' ', '_').replace('/', '_').lower()[:40]}",
        )
        pipeline.config.consensus.method = cfg["method"]
        pipeline.config.consensus.cross_model_models = cfg["models"]
        if cfg.get("k"):
            pipeline.config.sampling.k_samples = cfg["k"]

        metrics = pipeline.run()
        results[cfg["name"]] = metrics

        print(f"  mIoU = {metrics.get('mIoU', 'N/A'):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Exp6 Summary: Ablation Study")
    print("=" * 70)
    print(f"{'Configuration':<40s} {'mIoU':<10s} {'Δ':<10s}")
    print("-" * 60)

    full_miou = results.get("Full UGVU (PCU + UGCF + CMCF)", {}).get("mIoU", 0)
    for name, metrics in results.items():
        miou = metrics.get("mIoU", 0)
        delta = miou - full_miou
        print(f"{name:<40s} {miou:<10.4f} {delta:<+10.4f}")

    with open("outputs/exp6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    run_exp6(args.config)
