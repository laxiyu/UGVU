"""Expected Calibration Error (ECE) and related metrics.

ECE measures whether a model's confidence/uncertainty estimates are
well-calibrated — i.e., does "80% confidence" actually mean 80% accuracy?

For UGVU, we compute:
    - ECE: binned difference between confidence and accuracy.
    - MCE: maximum calibration error.
    - Reliability diagram data.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
    strategy: str = "uniform",
) -> dict:
    """Compute Expected Calibration Error.

    Args:
        confidences: (N,) float32 confidence values in [0, 1].
        accuracies: (N,) float32 binary accuracy values (1.0 = correct, 0.0 = wrong).
        num_bins: Number of bins.
        strategy: "uniform" (equal-width bins) or "quantile" (equal-count bins).

    Returns:
        Dict with:
            - ece: Expected Calibration Error (scalar).
            - mce: Maximum Calibration Error (scalar).
            - bin_confidences: Mean confidence per bin.
            - bin_accuracies: Mean accuracy per bin.
            - bin_counts: Number of samples per bin.
            - bin_edges: Bin boundary values.
    """
    N = len(confidences)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_confidences": [], "bin_accuracies": [], "bin_counts": [], "bin_edges": []}

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(confidences, np.linspace(0, 1, num_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bin_indices = np.digitize(confidences, bin_edges[1:-1], right=False)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for b in range(num_bins):
        mask = bin_indices == b
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            bin_confidences.append(confidences[mask].mean())
            bin_accuracies.append(accuracies[mask].mean())
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)

    # ECE = Σ (|B_b| / N) · |conf(B_b) - acc(B_b)|
    ece = np.sum((bin_counts / N) * np.abs(bin_confidences - bin_accuracies))

    # MCE = max_b |conf(B_b) - acc(B_b)|
    mce = np.max(np.abs(bin_confidences - bin_accuracies))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_confidences": bin_confidences.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def compute_ece_from_uncertainty(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    num_bins: int = 15,
    strategy: str = "uniform",
    confidence_from_uncertainty: bool = True,
) -> dict:
    """Compute ECE from per-pixel uncertainty and error maps.

    Args:
        uncertainty_map: (H, W) float32 uncertainty in [0, 1].
        error_map: (H, W) bool/int binary error map (1 = error, 0 = correct).
        num_bins: Number of bins.
        strategy: "uniform" or "quantile".
        confidence_from_uncertainty: If True, confidence = 1 - uncertainty.

    Returns:
        ECE result dict.
    """
    h, w = uncertainty_map.shape
    uncertainty_flat = uncertainty_map.ravel()
    error_flat = error_map.ravel().astype(np.float32)

    # Remove any NaN/Inf
    valid = np.isfinite(uncertainty_flat) & np.isfinite(error_flat)
    uncertainty_flat = uncertainty_flat[valid]
    error_flat = error_flat[valid]

    if confidence_from_uncertainty:
        confidences = 1.0 - uncertainty_flat  # higher uncertainty → lower confidence
    else:
        confidences = uncertainty_flat

    accuracies = 1.0 - error_flat  # error=1 → accuracy=0

    return compute_ece(confidences, accuracies, num_bins=num_bins, strategy=strategy)
