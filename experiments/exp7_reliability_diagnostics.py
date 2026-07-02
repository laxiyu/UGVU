"""Experiment 7: reliability diagnostics without remote API calls.

This offline experiment validates the reliability-analysis protocol proposed for
the paper. It creates complementary black-box model samples from the synthetic
Cityscapes fixture, then reports:
    - single-model vs CMCF accuracy,
    - within-model vs between-model uncertainty,
    - uncertainty/error alignment,
    - dense-prediction failure tags.

The goal is not to claim real API performance; it is a deterministic sanity
check for the diagnostic experiment machinery before running costly APIs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ugvu.consensus.consensus_fusion import consensus_fusion, fuse_cross_model
from ugvu.datasets.dataset import build_dataset
from ugvu.diagnostics import (
    aggregate_failure_tags,
    prediction_diagnostics,
    semantic_trivial_baselines,
    uncertainty_decomposition,
    uncertainty_error_alignment_summary,
)
from ugvu.metrics.metrics import evaluate_dataset
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map

OUT_DIR = Path("outputs/exp7_reliability_diagnostics")


def _corrupt_region(
    gt: np.ndarray,
    rng: np.random.Generator,
    region: np.ndarray,
    error_prob: float,
    num_classes: int,
) -> np.ndarray:
    pred = gt.copy()
    valid = (gt != 255) & region
    flip = valid & (rng.random(gt.shape) < error_prob)
    if flip.any():
        replacement = rng.integers(0, num_classes - 1, size=int(flip.sum()))
        pred_vals = pred[flip]
        replacement = replacement + (replacement >= pred_vals)
        pred[flip] = replacement
    return pred


def _simulate_complementary_model_stack(
    gt: np.ndarray,
    model_name: str,
    k: int,
    seed: int,
    num_classes: int,
) -> np.ndarray:
    """Create samples where two models are reliable in complementary regions."""
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    yy = np.arange(h)[:, None]
    top_half = yy < h // 2

    if model_name == "model_a":
        easy_region = top_half
    elif model_name == "model_b":
        easy_region = ~top_half
    else:
        easy_region = np.ones_like(gt, dtype=bool)

    stack = []
    for _ in range(k):
        pred = gt.copy()
        pred = _corrupt_region(pred, rng, easy_region, error_prob=0.04, num_classes=num_classes)
        pred = _corrupt_region(pred, rng, ~easy_region, error_prob=0.55, num_classes=num_classes)
        stack.append(pred)
    return np.stack(stack, axis=0).astype(np.int64)


def _run_image(gt: np.ndarray, image_index: int, k: int = 5, num_classes: int = 19) -> Dict:
    model_stacks = {
        "model_a": _simulate_complementary_model_stack(gt, "model_a", k, seed=1000 + image_index, num_classes=num_classes),
        "model_b": _simulate_complementary_model_stack(gt, "model_b", k, seed=2000 + image_index, num_classes=num_classes),
    }
    decomposition = uncertainty_decomposition(model_stacks, num_classes=num_classes, ignore_index=255, lambda_within=0.5)

    model_predictions = {}
    model_uncertainties = {}
    for model_name, stack in model_stacks.items():
        unc = build_uncertainty_map(
            stack,
            task="semantic_segmentation",
            uncertainty_type="entropy",
            num_classes=num_classes,
            ignore_index=255,
        )
        pred = consensus_fusion(stack, unc, method="majority", num_classes=num_classes, ignore_index=255)
        model_predictions[model_name] = pred
        model_uncertainties[model_name] = unc

    cmcf_pred = fuse_cross_model(
        model_predictions,
        model_uncertainties,
        num_classes=num_classes,
        ignore_index=255,
        temperature=0.25,
    )

    alignment = uncertainty_error_alignment_summary(
        {
            "within": decomposition["within_mean_map"],
            "between": decomposition["between_map"],
            "combined": decomposition["combined_map"],
        },
        cmcf_pred,
        gt,
        ignore_index=255,
    )

    per_prediction = {
        name: prediction_diagnostics(pred, gt, model_uncertainties[name], ignore_index=255, num_classes=num_classes)
        for name, pred in model_predictions.items()
    }
    per_prediction["cmcf"] = prediction_diagnostics(
        cmcf_pred,
        gt,
        decomposition["combined_map"],
        ignore_index=255,
        num_classes=num_classes,
    )

    return {
        "model_predictions": model_predictions,
        "cmcf_pred": cmcf_pred,
        "model_uncertainties": model_uncertainties,
        "combined_uncertainty": decomposition["combined_map"],
        "summary": {
            "image_index": image_index,
            "uncertainty_decomposition": decomposition["summary"],
            "uncertainty_error_alignment": alignment,
            "failure_tags": aggregate_failure_tags(per_prediction.values()),
            "per_prediction_diagnostics": per_prediction,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset("cityscapes", root="data/synthetic_cityscapes", split="val", max_samples=8)
    gts = [dataset[i]["label"].astype(np.int64) for i in range(len(dataset))]

    model_preds = {"model_a": [], "model_b": [], "cmcf": []}
    model_uncs = {"model_a": [], "model_b": [], "cmcf": []}
    summaries = []

    for idx, gt in enumerate(gts):
        item = _run_image(gt, idx)
        for model_name in ("model_a", "model_b"):
            model_preds[model_name].append(item["model_predictions"][model_name])
            model_uncs[model_name].append(item["model_uncertainties"][model_name])
        model_preds["cmcf"].append(item["cmcf_pred"])
        model_uncs["cmcf"].append(item["combined_uncertainty"])
        summaries.append(item["summary"])

    metrics = {
        name: evaluate_dataset(preds, gts, task="semantic_segmentation", num_classes=19, ignore_index=255)
        for name, preds in model_preds.items()
    }
    diag_per = [prediction_diagnostics(pred, gt, unc, ignore_index=255, num_classes=19)
                for pred, gt, unc in zip(model_preds["cmcf"], gts, model_uncs["cmcf"])]

    report = {
        "dataset": "synthetic_cityscapes",
        "protocol": "offline_complementary_model_diagnostic",
        "note": "Deterministic validation of the diagnostic protocol, not real API performance.",
        "metrics": metrics,
        "cmcf_delta_mIoU_vs_best_single": float(
            metrics["cmcf"]["mIoU"] - max(metrics["model_a"]["mIoU"], metrics["model_b"]["mIoU"])
        ),
        "diagnostics": {
            "cmcf_failure_tag_counts": aggregate_failure_tags(diag_per),
            "trivial_baselines": semantic_trivial_baselines(gts, num_classes=19, ignore_index=255),
            "per_image": summaries,
        },
    }

    with (OUT_DIR / "reliability_diagnostic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

