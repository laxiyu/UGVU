"""Uncertainty-Guided Consensus Fusion (UGCF).

Unlike simple majority voting, UGCF weights each pixel's vote by the
inverse of its per-sample uncertainty, giving more influence to
confident predictions.

Core idea:
    Low uncertainty  → high weight
    High uncertainty → low weight

The weight for sample k at pixel (i,j) is:
    w_k(i,j) = softmax_k( -U_k(i,j) / τ )

where U_k is the per-sample uncertainty map and τ is a temperature.
"""

from __future__ import annotations

import numpy as np


def uncertainty_guided_fusion(
    mask_stack: np.ndarray,
    uncertainty_maps: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    temperature: float = 1.0,
) -> np.ndarray:
    """UGCF: fuse K masks weighted by per-sample per-pixel uncertainty.

    Args:
        mask_stack: (K, H, W) int64 class predictions.
        uncertainty_maps: (K, H, W) float32 per-sample uncertainty.
                          Higher = more uncertain → lower weight.
        num_classes: Number of classes.
        ignore_index: Ignore label.
        temperature: Softmax temperature τ. Lower τ → sharper weighting.

    Returns:
        (H, W) int64 fused mask.

    Formula:
        For each pixel p:
            weight_k(p) = softmax_k( -uncertainty_k(p) / τ )
            final_class(p) = argmax_c Σ_k weight_k(p) · 1[mask_k(p) == c]
    """
    K, H, W = mask_stack.shape

    # Normalize uncertainty to [0, 1] if not already
    u_min = uncertainty_maps.min()
    u_max = uncertainty_maps.max()
    if u_max - u_min > 1e-6:
        u_norm = (uncertainty_maps - u_min) / (u_max - u_min)  # (K, H, W)
    else:
        u_norm = uncertainty_maps

    # Confidence = 1 - uncertainty
    confidence = 1.0 - u_norm  # (K, H, W)

    # Softmax over K dimension
    logits = confidence / max(temperature, 1e-6)  # (K, H, W)
    # Stable softmax
    logits_max = logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    weights = exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-8)  # (K, H, W)

    # Weighted vote
    vote = np.zeros((num_classes, H, W), dtype=np.float32)
    for k in range(K):
        valid = mask_stack[k] != ignore_index
        for c in range(num_classes):
            match = valid & (mask_stack[k] == c)
            vote[c][match] += weights[k][match]

    result = vote.argmax(axis=0).astype(np.int64)
    # Where no valid votes → ignore
    total = vote.sum(axis=0)
    result[total == 0] = ignore_index

    return result


def uncertainty_guided_fusion_continuous(
    value_stack: np.ndarray,
    uncertainty_maps: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """UGCF for continuous predictions (e.g. depth).

    Args:
        value_stack: (K, H, W) float32 predictions.
        uncertainty_maps: (K, H, W) float32 per-sample uncertainty.
        temperature: Softmax temperature.

    Returns:
        (H, W) float32 weighted average.
    """
    K = value_stack.shape[0]

    # Normalize uncertainty
    u_min = uncertainty_maps.min()
    u_max = uncertainty_maps.max()
    if u_max - u_min > 1e-6:
        u_norm = (uncertainty_maps - u_min) / (u_max - u_min)
    else:
        u_norm = uncertainty_maps

    confidence = 1.0 - u_norm
    logits = confidence / max(temperature, 1e-6)
    logits_max = logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    weights = exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-8)

    return (value_stack * weights).sum(axis=0).astype(np.float32)
