"""Reliability diagrams — calibration curve plotting data.

A reliability diagram plots expected confidence vs. observed accuracy.
A perfectly calibrated model lies on the y=x diagonal.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def reliability_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Compute reliability curve data for plotting.

    Args:
        confidences: (N,) float32 confidence values.
        accuracies: (N,) float32 binary accuracy (1=correct, 0=wrong).
        num_bins: Number of bins.
        strategy: "uniform" or "quantile".

    Returns:
        Dict with:
            - mean_predicted: Mean confidence per bin (x-axis).
            - mean_observed: Mean accuracy per bin (y-axis).
            - bin_counts: Sample count per bin.
            - perfect_line: y=x diagonal reference.
    """
    from .ece import compute_ece
    ece_result = compute_ece(confidences, accuracies, num_bins=num_bins, strategy=strategy)

    bin_conf = np.array(ece_result["bin_confidences"])
    bin_acc = np.array(ece_result["bin_accuracies"])
    bin_counts = np.array(ece_result["bin_counts"])

    # Only keep non-empty bins
    non_empty = bin_counts > 0
    x = np.linspace(0, 1, 100)

    return {
        "mean_predicted": bin_conf[non_empty],
        "mean_observed": bin_acc[non_empty],
        "bin_counts": bin_counts[non_empty],
        "perfect_x": x,
        "perfect_y": x,
    }


def reliability_from_maps(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    num_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Compute reliability curve data from uncertainty and error maps.

    Args:
        uncertainty_map: (H, W) float32.
        error_map: (H, W) bool/int.

    Returns:
        Dict ready for plotting.
    """
    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.float32)
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    confidences = 1.0 - u_flat[valid]
    accuracies = 1.0 - e_flat[valid]
    return reliability_curve(confidences, accuracies, num_bins=num_bins, strategy=strategy)
