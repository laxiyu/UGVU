"""Evaluation metrics for UGVU tasks.

Supported metrics:
    - Semantic segmentation: mIoU, IoU per class, pixel accuracy
    - Referring segmentation: cIoU (cumulative IoU), overall IoU, precision/recall
    - Depth estimation: AbsRel, δ1, RMSE, RMSE log, SqRel
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ============================================================================
# Semantic Segmentation
# ============================================================================

def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute (num_classes, num_classes) confusion matrix.

    Rows = ground truth, Columns = predictions.
    """
    mask = (target != ignore_index) & (pred != ignore_index)
    pred_valid = pred[mask]
    target_valid = target[mask]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (target_valid, pred_valid), 1)
    return cm


def compute_iou(conf_mat: np.ndarray) -> np.ndarray:
    """Compute per-class IoU from confusion matrix."""
    intersection = np.diag(conf_mat)
    union = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - intersection
    iou = np.zeros_like(intersection, dtype=np.float32)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    ious = compute_iou(cm)
    # Only average over classes present in ground truth or predictions
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    if present.sum() == 0:
        return 0.0
    return float(ious[present].mean())


def per_class_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[int, float]:
    """Compute IoU for each class."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    ious = compute_iou(cm)
    return {c: float(ious[c]) for c in range(num_classes)}


def pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """Pixel-wise accuracy."""
    mask = (target != ignore_index) & (pred != ignore_index)
    if mask.sum() == 0:
        return 0.0
    return float((pred[mask] == target[mask]).mean())


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Compute all semantic segmentation metrics.

    Returns:
        Dict with mIoU, pixel_accuracy, and per-class IoU dict.
    """
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    ious = compute_iou(cm)
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0

    return {
        "mIoU": float(ious[present].mean()) if present.any() else 0.0,
        "pixel_accuracy": pixel_accuracy(pred, target, ignore_index),
        "per_class_iou": {c: float(ious[c]) for c in range(num_classes)},
    }


# ============================================================================
# Referring Segmentation
# ============================================================================

def cumulative_iou(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute cumulative IoU for referring segmentation.

    cIoU = TP / (TP + FP + FN) = intersection / union
    Treats the task as binary (foreground=1, background=0).
    """
    pred_bin = (pred > 0).astype(np.int64)
    target_bin = (target > 0).astype(np.int64)

    intersection = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()

    if union == 0:
        return 1.0  # both empty → perfect match
    return float(intersection / union)


def overall_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Alias for cumulative IoU."""
    return cumulative_iou(pred, target)


def precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for binary referring segmentation."""
    pred_bin = (pred > 0).astype(np.int64)
    target_bin = (target > 0).astype(np.int64)

    tp = (pred_bin & target_bin).sum()
    fp = (pred_bin & ~target_bin).sum()
    fn = (~pred_bin & target_bin).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ============================================================================
# Depth Estimation
# ============================================================================

def depth_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    median_scale: bool = True,
) -> Dict[str, float]:
    """Compute standard depth estimation metrics.

    Args:
        pred: (H, W) float32 predicted depth.
        target: (H, W) float32 ground truth depth.
        valid_mask: Optional (H, W) bool mask of valid pixels.
        median_scale: If True, apply median scaling to align prediction scale.

    Returns:
        Dict with AbsRel, SqRel, RMSE, RMSE_log, δ1, δ2, δ3.
    """
    if valid_mask is None:
        valid_mask = (target > 0) & np.isfinite(target)

    p = pred[valid_mask].astype(np.float64)
    t = target[valid_mask].astype(np.float64)

    if len(p) == 0:
        return {"AbsRel": float("nan"), "SqRel": float("nan"), "RMSE": float("nan"),
                "RMSE_log": float("nan"), "delta1": float("nan"), "delta2": float("nan"), "delta3": float("nan")}

    # Median scaling
    if median_scale:
        scale = np.median(t) / np.median(p)
        p = p * scale

    # Clip to avoid log(0)
    eps = 1e-6
    p = np.maximum(p, eps)
    t = np.maximum(t, eps)

    # Absolute relative difference
    abs_rel = np.mean(np.abs(t - p) / t)

    # Squared relative difference
    sq_rel = np.mean((t - p) ** 2 / t)

    # RMSE
    rmse = np.sqrt(np.mean((t - p) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(t) - np.log(p)) ** 2))

    # Threshold accuracy
    ratio = np.maximum(t / p, p / t)
    delta1 = float((ratio < 1.25).mean())
    delta2 = float((ratio < 1.25 ** 2).mean())
    delta3 = float((ratio < 1.25 ** 3).mean())

    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "RMSE_log": float(rmse_log),
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


# ============================================================================
# Batch evaluation
# ============================================================================

def evaluate_dataset(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    task: str = "semantic_segmentation",
    num_classes: int = 19,
    ignore_index: int = 255,
) -> Dict:
    """Evaluate a list of predictions against ground truths.

    Args:
        predictions: List of (H, W) prediction arrays.
        ground_truths: List of (H, W) ground truth arrays.
        task: "semantic_segmentation", "referring_segmentation", or "depth_estimation".
        num_classes: For semantic segmentation.
        ignore_index: For semantic segmentation.

    Returns:
        Dict with aggregated metrics.
    """
    if task == "depth_estimation":
        all_metrics = []
        for pred, gt in zip(predictions, ground_truths):
            all_metrics.append(depth_metrics(pred, gt))

        # Average across images
        keys = ["AbsRel", "SqRel", "RMSE", "RMSE_log", "delta1", "delta2", "delta3"]
        agg = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        return agg

    elif task == "referring_segmentation":
        cious = []
        precisions = []
        recalls = []
        for pred, gt in zip(predictions, ground_truths):
            cious.append(cumulative_iou(pred, gt))
            prf = precision_recall_f1(pred, gt)
            precisions.append(prf["precision"])
            recalls.append(prf["recall"])

        mean_p = np.mean(precisions)
        mean_r = np.mean(recalls)
        return {
            "cIoU": float(np.mean(cious)),
            "Overall_IoU": float(np.mean(cious)),
            "Precision": float(mean_p),
            "Recall": float(mean_r),
            "F1": float(2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0,
        }

    else:  # semantic_segmentation
        total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, gt in zip(predictions, ground_truths):
            total_cm += confusion_matrix(pred, gt, num_classes, ignore_index)

        ious = compute_iou(total_cm)
        present = (total_cm.sum(axis=1) + total_cm.sum(axis=0)) > 0

        return {
            "mIoU": float(ious[present].mean()) if present.any() else 0.0,
            "per_class_iou": {c: float(ious[c]) for c in range(num_classes)},
        }
