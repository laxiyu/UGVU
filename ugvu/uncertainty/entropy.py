"""Pixel-wise entropy for semantic segmentation uncertainty.

Computes H(p) = -p·log(p) - (1-p)·log(1-p) per pixel
across K generated samples.
"""

from __future__ import annotations

import numpy as np


def pixelwise_entropy(
    mask_stack: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-pixel Shannon entropy from K categorical masks.

    Args:
        mask_stack: (K, H, W) int64 array of class predictions.
        num_classes: Total number of classes C.
        ignore_index: Pixels with this value are excluded and get entropy = 0.
        eps: Small constant for numerical stability in log.

    Returns:
        (H, W) float32 entropy map. Higher = more disagreement across samples.

    Formula:
        For each pixel (i, j):
            p_c = count(sample[:, i, j] == c) / K
            H(i,j) = -Σ_c p_c · log(p_c)

    Interpretation:
        - H ≈ 0: all K samples agree → high confidence
        - H ≈ log(num_classes): samples disagree uniformly → high uncertainty
    """
    K, H, W = mask_stack.shape
    entropy_map = np.zeros((H, W), dtype=np.float32)

    for c in range(num_classes):
        # Fraction of samples predicting class c at each pixel
        p_c = (mask_stack == c).mean(axis=0).astype(np.float32)
        # Only add where p_c > 0
        mask = p_c > 0
        entropy_map[mask] -= p_c[mask] * np.log(p_c[mask] + eps)

    # Zero out ignore pixels
    # (Any pixel that is ignore in ALL samples — but we treat partial-ignore as valid)
    all_ignore = (mask_stack == ignore_index).all(axis=0)
    entropy_map[all_ignore] = 0.0

    # Normalize to [0, 1] by dividing by max possible entropy
    max_entropy = np.log(num_classes)
    if max_entropy > 0:
        entropy_map /= max_entropy

    return entropy_map.astype(np.float32)


def pixelwise_dirichlet_entropy(
    mask_stack: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute Dirichlet-smoothed pixel-wise entropy for small-K regimes.

    This is a black-box adaptation of predictive entropy when only a few
    repeated samples are available. It adds a symmetric Dirichlet prior to the
    empirical label counts, reducing the high variance of raw frequency
    estimates when K is small.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    K, H, W = mask_stack.shape
    entropy_map = np.zeros((H, W), dtype=np.float32)
    valid_k = (mask_stack != ignore_index).sum(axis=0).astype(np.float32)
    denom = valid_k + float(alpha) * num_classes

    for c in range(num_classes):
        counts = (mask_stack == c).sum(axis=0).astype(np.float32)
        p_c = (counts + float(alpha)) / np.maximum(denom, eps)
        mask = denom > eps
        entropy_map[mask] -= p_c[mask] * np.log(p_c[mask] + eps)

    all_ignore = (mask_stack == ignore_index).all(axis=0)
    entropy_map[all_ignore] = 0.0

    max_entropy = np.log(num_classes)
    if max_entropy > 0:
        entropy_map /= max_entropy

    return np.clip(entropy_map, 0.0, 1.0).astype(np.float32)


def pixelwise_entropy_continuous(
    value_stack: np.ndarray,
    num_bins: int = 50,
    value_range: tuple = (0.0, 10.0),
) -> np.ndarray:
    """Compute per-pixel entropy for continuous-valued predictions (e.g. depth).

    Discretizes the continuous values into bins and computes entropy on the
    histogram per pixel.

    Args:
        value_stack: (K, H, W) float32 array of continuous predictions.
        num_bins: Number of bins for discretization.
        value_range: (min, max) range for binning.

    Returns:
        (H, W) float32 entropy map normalized to [0, 1].
    """
    K, H, W = value_stack.shape
    vmin, vmax = value_range
    bins = np.linspace(vmin, vmax, num_bins + 1)

    entropy_map = np.zeros((H, W), dtype=np.float32)

    for b in range(num_bins):
        in_bin = (value_stack >= bins[b]) & (value_stack < bins[b + 1])
        p_b = in_bin.mean(axis=0).astype(np.float32)
        mask = p_b > 0
        entropy_map[mask] -= p_b[mask] * np.log(p_b[mask] + 1e-8)

    max_entropy = np.log(num_bins)
    if max_entropy > 0:
        entropy_map /= max_entropy

    return entropy_map.astype(np.float32)
