"""Diagnostic utilities for reliability-focused UGVU experiments.

These helpers deliberately complement standard accuracy metrics. They make it
harder to overclaim low absolute mIoU results, and easier to report what the
experiment actually probes: calibration, collapse, uncertainty/error alignment,
and relative gains from repeated consensus.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .metrics.metrics import evaluate_dataset, segmentation_metrics


def finite_float(value: Any) -> Optional[float]:
    """Convert numeric values to JSON-safe floats, mapping NaN/inf to None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def relative_change(new: Any, old: Any) -> Optional[float]:
    """Return (new - old) / abs(old), or None when unavailable."""
    new_f = finite_float(new)
    old_f = finite_float(old)
    if new_f is None or old_f is None or abs(old_f) < 1e-12:
        return None
    return (new_f - old_f) / abs(old_f)


def semantic_trivial_baselines(
    ground_truths: List[np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute simple dense-prediction baselines from labels only.

    The baselines are diagnostic references, not competing methods:
    - global_majority: predict the most frequent GT class for all pixels.
    - image_majority_oracle: predict each image's most frequent GT class.
    - uniform_random: sample valid classes uniformly with a fixed seed.
    """
    valid_pixels = []
    for gt in ground_truths:
        vals = gt[gt != ignore_index].astype(np.int64)
        if vals.size:
            valid_pixels.append(vals)
    if not valid_pixels:
        return {}

    all_valid = np.concatenate(valid_pixels)
    counts = np.bincount(all_valid, minlength=num_classes)
    global_majority = int(np.argmax(counts))
    present_classes = np.flatnonzero(counts > 0)
    rng = np.random.default_rng(seed)

    global_preds = []
    image_oracle_preds = []
    random_preds = []
    for gt in ground_truths:
        valid = gt != ignore_index
        global_pred = np.full(gt.shape, ignore_index, dtype=np.int64)
        global_pred[valid] = global_majority
        global_preds.append(global_pred)

        vals = gt[valid].astype(np.int64)
        if vals.size:
            img_counts = np.bincount(vals, minlength=num_classes)
            img_majority = int(np.argmax(img_counts))
        else:
            img_majority = global_majority
        oracle_pred = np.full(gt.shape, ignore_index, dtype=np.int64)
        oracle_pred[valid] = img_majority
        image_oracle_preds.append(oracle_pred)

        random_pred = np.full(gt.shape, ignore_index, dtype=np.int64)
        if present_classes.size:
            random_pred[valid] = rng.choice(present_classes, size=int(valid.sum()))
        random_preds.append(random_pred)

    return {
        "global_majority": {
            "description": "Predict the dataset-level majority class everywhere.",
            "class_id": global_majority,
            "metrics": evaluate_dataset(global_preds, ground_truths, "semantic_segmentation", num_classes, ignore_index),
        },
        "image_majority_oracle": {
            "description": "Predict each image's GT-majority class everywhere; diagnostic oracle for class collapse.",
            "metrics": evaluate_dataset(image_oracle_preds, ground_truths, "semantic_segmentation", num_classes, ignore_index),
        },
        "uniform_random": {
            "description": "Uniformly sample among classes present in the evaluation labels.",
            "seed": seed,
            "metrics": evaluate_dataset(random_preds, ground_truths, "semantic_segmentation", num_classes, ignore_index),
        },
    }


def prediction_diagnostics(
    pred: np.ndarray,
    gt: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    ignore_index: int = 255,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize dense-prediction failure modes for one image."""
    valid_gt = gt != ignore_index
    valid_pred = (pred != ignore_index) & valid_gt
    pred_vals = pred[valid_pred].astype(np.int64)
    gt_vals = gt[valid_gt].astype(np.int64)

    diag: Dict[str, Any] = {
        "valid_pixels": int(valid_gt.sum()),
        "pred_valid_pixels": int(valid_pred.sum()),
        "ignore_rate": finite_float(1.0 - (valid_pred.sum() / max(1, valid_gt.sum()))),
        "pred_num_classes": int(np.unique(pred_vals).size) if pred_vals.size else 0,
        "gt_num_classes": int(np.unique(gt_vals).size) if gt_vals.size else 0,
        "dominant_pred_class": None,
        "dominant_pred_fraction": None,
        "error_rate": None,
        "pixel_accuracy": None,
        "mIoU": None,
        "failure_tags": [],
    }

    if pred_vals.size:
        classes, counts = np.unique(pred_vals, return_counts=True)
        top_idx = int(np.argmax(counts))
        diag["dominant_pred_class"] = int(classes[top_idx])
        diag["dominant_pred_fraction"] = finite_float(counts[top_idx] / pred_vals.size)

    if num_classes is not None and valid_gt.any():
        metrics = segmentation_metrics(pred, gt, num_classes=num_classes, ignore_index=ignore_index)
        diag["mIoU"] = finite_float(metrics.get("mIoU"))
        diag["pixel_accuracy"] = finite_float(metrics.get("pixel_accuracy"))

    if valid_gt.any():
        error_map = (pred != gt) & valid_gt
        diag["error_rate"] = finite_float(error_map.mean())

        if uncertainty is not None and uncertainty.shape[:2] == gt.shape[:2]:
            unc = uncertainty.astype(np.float32)
            err_unc = unc[error_map]
            ok_unc = unc[(~error_map) & valid_gt]
            diag["mean_uncertainty_error"] = finite_float(err_unc.mean()) if err_unc.size else None
            diag["mean_uncertainty_correct"] = finite_float(ok_unc.mean()) if ok_unc.size else None
            if err_unc.size and ok_unc.size:
                diag["uncertainty_error_gap"] = finite_float(err_unc.mean() - ok_unc.mean())
            else:
                diag["uncertainty_error_gap"] = None

    tags = []
    if diag["dominant_pred_fraction"] is not None and diag["dominant_pred_fraction"] >= 0.80:
        tags.append("class_collapse")
    if diag["pred_num_classes"] <= 2 and diag["gt_num_classes"] >= 4:
        tags.append("low_class_diversity")
    if diag["ignore_rate"] is not None and diag["ignore_rate"] >= 0.20:
        tags.append("high_ignore_rate")
    if diag["error_rate"] is not None and diag["error_rate"] >= 0.80:
        tags.append("high_error_rate")
    if diag.get("uncertainty_error_gap") is not None and diag["uncertainty_error_gap"] <= 0:
        tags.append("uncertainty_misaligned")
    diag["failure_tags"] = tags
    return diag


def aggregate_failure_tags(per_image: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for item in per_image:
        for tag in item.get("failure_tags", []):
            counts[tag] += 1
    return dict(sorted(counts.items()))


def attach_k_sweep_summary(results: List[Dict[str, Any]], metric_keys: Iterable[str] = ("mIoU", "ECE", "AUROC", "Spearman")) -> Dict[str, Any]:
    """Summarize K=1 -> larger-K changes per model for paper tables."""
    by_model: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for item in results:
        if item.get("ok", True) is False:
            continue
        model = item.get("model")
        k = item.get("k")
        if model is None or k is None:
            continue
        by_model[str(model)][int(k)] = item

    summary = {}
    for model, cases in by_model.items():
        if 1 not in cases:
            continue
        base = cases[1].get("metrics", {})
        model_summary = {}
        for k, item in sorted(cases.items()):
            if k == 1:
                continue
            metrics = item.get("metrics", {})
            deltas = {}
            for key in metric_keys:
                deltas[f"{key}_absolute_change"] = finite_float(metrics.get(key)) - finite_float(base.get(key)) if finite_float(metrics.get(key)) is not None and finite_float(base.get(key)) is not None else None
                deltas[f"{key}_relative_change"] = relative_change(metrics.get(key), base.get(key))
            model_summary[f"k{k}_vs_k1"] = deltas
        if model_summary:
            summary[model] = model_summary
    return summary


def uncertainty_decomposition(
    all_model_masks: Dict[str, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    lambda_within: float = 0.5,
) -> Dict[str, Any]:
    """Decompose black-box consensus uncertainty into within/between parts.

    Within-model uncertainty measures sample disagreement for each model. Between-
    model uncertainty measures disagreement among the per-model consensus masks.
    This directly supports the CMCF framing in the paper: stochastic instability
    and cross-endpoint disagreement are different reliability signals.
    """
    if not all_model_masks:
        raise ValueError("all_model_masks must contain at least one model")

    from .consensus.majority_vote import majority_vote
    from .uncertainty.uncertainty_map import build_uncertainty_map

    lambda_within = float(np.clip(lambda_within, 0.0, 1.0))
    model_names = list(all_model_masks.keys())
    within_maps: Dict[str, np.ndarray] = {}
    model_consensus: Dict[str, np.ndarray] = {}

    for model_name in model_names:
        stack = all_model_masks[model_name]
        if stack.ndim != 3:
            raise ValueError(f"Expected (K,H,W) stack for {model_name}, got {stack.shape}")
        within_maps[model_name] = build_uncertainty_map(
            stack,
            task="semantic_segmentation",
            uncertainty_type="entropy",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        model_consensus[model_name] = majority_vote(stack, ignore_index=ignore_index)

    within_stack = np.stack([within_maps[m] for m in model_names], axis=0)
    within_mean = within_stack.mean(axis=0).astype(np.float32)
    consensus_stack = np.stack([model_consensus[m] for m in model_names], axis=0)
    between = build_uncertainty_map(
        consensus_stack,
        task="semantic_segmentation",
        uncertainty_type="entropy",
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    combined = (lambda_within * within_mean + (1.0 - lambda_within) * between).astype(np.float32)

    valid = ~(consensus_stack == ignore_index).all(axis=0)
    disagree = np.zeros_like(valid, dtype=bool)
    if len(model_names) > 1:
        first = consensus_stack[0]
        disagree = np.any(consensus_stack != first[None, ...], axis=0) & valid

    return {
        "within_maps": within_maps,
        "within_mean_map": within_mean,
        "between_map": between.astype(np.float32),
        "combined_map": combined,
        "model_consensus": model_consensus,
        "summary": {
            "mean_within_uncertainty": finite_float(within_mean[valid].mean()) if valid.any() else None,
            "mean_between_uncertainty": finite_float(between[valid].mean()) if valid.any() else None,
            "mean_combined_uncertainty": finite_float(combined[valid].mean()) if valid.any() else None,
            "model_disagreement_rate": finite_float(disagree.sum() / max(1, valid.sum())),
            "lambda_within": lambda_within,
        },
    }


def uncertainty_error_alignment_summary(
    uncertainty_maps: Dict[str, np.ndarray],
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Summarize how each uncertainty signal aligns with pixel errors."""
    from .calibration.correlation import correlation_report

    valid = gt != ignore_index
    error_map = (pred != gt) & valid
    report = {}
    for name, uncertainty in uncertainty_maps.items():
        corr = correlation_report(uncertainty[valid], error_map[valid]) if valid.any() else {
            "spearman_r": 0.0,
            "pearson_r": 0.0,
            "auroc": 0.5,
        }
        err_unc = uncertainty[error_map]
        ok_unc = uncertainty[(~error_map) & valid]
        report[name] = {
            "mean_uncertainty_error": finite_float(err_unc.mean()) if err_unc.size else None,
            "mean_uncertainty_correct": finite_float(ok_unc.mean()) if ok_unc.size else None,
            "uncertainty_error_gap": finite_float(err_unc.mean() - ok_unc.mean()) if err_unc.size and ok_unc.size else None,
            "spearman_r": finite_float(corr.get("spearman_r")),
            "pearson_r": finite_float(corr.get("pearson_r")),
            "auroc": finite_float(corr.get("auroc")),
        }
    return report




