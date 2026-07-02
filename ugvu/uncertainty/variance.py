"""Pixel-wise variance for continuous predictions (e.g. depth).

Computes per-pixel variance across K generated samples.
"""

from __future__ import annotations

import numpy as np


def pixelwise_variance(value_stack: np.ndarray) -> np.ndarray:
    """Compute per-pixel variance across K samples.

    Args:
        value_stack: (K, H, W) or (K, H, W, C) array.

    Returns:
        (H, W) float32 variance map.
    """
    if value_stack.ndim == 4:
        # (K, H, W, C) 鈥?compute variance per channel then average
        var_per_channel = value_stack.var(axis=0)  # (H, W, C)
        return var_per_channel.mean(axis=-1).astype(np.float32)
    else:
        # (K, H, W)
        return value_stack.var(axis=0).astype(np.float32)


def pixelwise_std(value_stack: np.ndarray) -> np.ndarray:
    """Compute per-pixel standard deviation across K samples."""
    return np.sqrt(pixelwise_variance(value_stack))


def pixelwise_cv(value_stack: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute per-pixel coefficient of variation (std / mean).

    Useful for depth estimation where absolute variance depends on depth scale.
    """
    mean = value_stack.mean(axis=0)
    std = np.sqrt(pixelwise_variance(value_stack))
    return std / (np.abs(mean) + eps)


def agreement_ratio(mask_stack: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Compute per-pixel agreement ratio 鈥?fraction of samples matching the mode.

    A simple alternative to entropy: higher ratio = more consensus.

    Args:
        mask_stack: (K, H, W) int64 class predictions.

    Returns:
        (H, W) float32 agreement ratio in [0, 1].
    """
    from ugvu.consensus.majority_vote import majority_vote

    mode = majority_vote(mask_stack, ignore_index=ignore_index)
    agreement = (mask_stack == mode).mean(axis=0).astype(np.float32)
    all_ignore = (mask_stack == ignore_index).all(axis=0)
    agreement[all_ignore] = 1.0
    return agreement

