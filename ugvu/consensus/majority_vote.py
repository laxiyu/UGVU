"""Majority vote baseline 鈥?simple hard voting across K samples.

For each pixel, the class that appears most frequently across the K samples
is selected as the final prediction.
"""

from __future__ import annotations

import numpy as np


def majority_vote(mask_stack: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Simple majority (mode) voting across K categorical masks.

    Args:
        mask_stack: (K, H, W) int64 class predictions.
        ignore_index: Value to treat as "no vote".

    Returns:
        (H, W) int64 final mask (mode per pixel).
        Ties are broken by choosing the smaller class index.
    """
    K, H, W = mask_stack.shape

    if K == 1:
        return mask_stack[0].copy()

    result = np.full((H, W), ignore_index, dtype=np.int64)
    best_count = np.zeros((H, W), dtype=np.int16)

    for label in np.sort(np.unique(mask_stack)):
        label = int(label)
        if label == ignore_index:
            continue
        count = np.sum(mask_stack == label, axis=0)
        update = count > best_count
        result[update] = label
        best_count[update] = count[update]

    return result


def majority_vote_weighted(
    mask_stack: np.ndarray,
    weights: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Weighted majority vote 鈥?weights are per-sample scalars.

    Args:
        mask_stack: (K, H, W) int64.
        weights: (K,) float32 weights per sample.
        num_classes: Number of classes.
        ignore_index: Ignore label.

    Returns:
        (H, W) int64 final mask.
    """
    K, H, W = mask_stack.shape
    # Build (K, H, W) 鈫?(C, H, W) one-hot
    one_hot = np.zeros((num_classes, H, W), dtype=np.float32)
    for k in range(K):
        valid = mask_stack[k] != ignore_index
        for c in range(num_classes):
            one_hot[c][valid & (mask_stack[k] == c)] += weights[k]

    result = one_hot.argmax(axis=0).astype(np.int64)
    # Where all classes have zero votes 鈫?ignore
    total_votes = one_hot.sum(axis=0)
    result[total_votes == 0] = ignore_index

    return result


