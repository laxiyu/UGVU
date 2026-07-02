"""Experiment 5: Calibration Analysis.

Analyzes whether uncertainty predicts actual error:
    - Spearman correlation (ρ)
    - ECE (Expected Calibration Error)
    - Reliability diagrams
    - AUROC (uncertainty → error prediction)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from ugvu.pipeline import UGVUPipeline


def run_exp5(config_path: str = "configs/default.yaml"):
    """Run Exp5: Calibration analysis."""
    print(f"\n{'='*60}")
    print("Exp5: Calibration Analysis")
    print(f"{'='*60}")

    pipeline = UGVUPipeline(
        config_path,
        task="semantic_segmentation",
        output_dir="outputs/exp5_calibration",
    )

    # Run full pipeline first
    pipeline.run()

    # Run calibration
    calib_result = pipeline.run_calibration()

    print(f"\n{'='*60}")
    print("Exp5 Results: Calibration")
    print(f"{'='*60}")
    print(f"  ECE (Expected Calibration Error): {calib_result['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error):  {calib_result['mce']:.4f}")
    print(f"  Spearman ρ (Uncertainty↔Error):   {calib_result['spearman_r']:.4f}")
    print(f"  AUROC (Error prediction):          {calib_result['auroc']:.4f}")

    # Interpretation
    spearman = calib_result["spearman_r"]
    if spearman > 0.3:
        interp = "Strong: the model KNOWS when it's wrong!"
    elif spearman > 0.1:
        interp = "Moderate: uncertainty somewhat correlates with error."
    else:
        interp = "Weak: uncertainty is not a good error predictor."
    print(f"\n  Interpretation: {interp}")

    return calib_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()
    run_exp5(args.config)
