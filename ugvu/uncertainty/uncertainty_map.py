"""Unified uncertainty map builder — entry point for PCU.

Builds a Pixel-wise Consensus Uncertainty (PCU) map from a stack of
K generated samples, choosing the appropriate metric based on task type.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .entropy import pixelwise_entropy, pixelwise_dirichlet_entropy, pixelwise_entropy_continuous
from .variance import pixelwise_variance, pixelwise_std, agreement_ratio


UncertaintyType = Literal["entropy", "dirichlet_entropy", "variance", "std", "agreement", "entropy_continuous"]


def build_uncertainty_map(
    prediction_stack: np.ndarray,
    task: str = "semantic_segmentation",
    uncertainty_type: UncertaintyType = "entropy",
    num_classes: int = 19,
    ignore_index: int = 255,
    value_range: tuple = (0.0, 10.0),
    **kwargs,
) -> np.ndarray:
    """Build a pixel-wise uncertainty map from K predictions.

    This is the primary entry point for PCU (Pixel-wise Consensus Uncertainty).

    Args:
        prediction_stack: (K, H, W) int64 for segmentation, or (K, H, W) float32 for depth.
        task: "semantic_segmentation", "referring_segmentation", or "depth_estimation".
        uncertainty_type: Which uncertainty metric to use.
        num_classes: Number of classes (for entropy calculation).
        ignore_index: Ignore label index.
        value_range: (min, max) for continuous entropy binning.

    Returns:
        (H, W) float32 uncertainty map. Values are normalized to [0, 1] where possible.

    Usage:
        >>> masks = np.stack([mask1, mask2, mask3, mask4, mask5])  # (5, H, W)
        >>> uncertainty = build_uncertainty_map(masks, task="semantic_segmentation")
        >>> # uncertainty is (H, W) float32, 0=low uncertainty, 1=high uncertainty
    """
    is_continuous = task == "depth_estimation"

    if uncertainty_type == "entropy":
        if is_continuous:
            return pixelwise_entropy_continuous(
                prediction_stack,
                num_bins=kwargs.get("num_bins", 50),
                value_range=value_range,
            )
        else:
            return pixelwise_entropy(
                prediction_stack,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

    elif uncertainty_type == "entropy_continuous":
        return pixelwise_entropy_continuous(
            prediction_stack,
            num_bins=kwargs.get("num_bins", 50),
            value_range=value_range,
        )

    elif uncertainty_type == "dirichlet_entropy":
        return pixelwise_dirichlet_entropy(
            prediction_stack,
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=kwargs.get("alpha", 0.5),
        )

    elif uncertainty_type == "variance":
        return pixelwise_variance(prediction_stack)

    elif uncertainty_type == "std":
        return pixelwise_std(prediction_stack)

    elif uncertainty_type == "agreement":
        return 1.0 - agreement_ratio(prediction_stack, ignore_index=ignore_index)

    else:
        raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")


def uncertainty_to_confidence(uncertainty_map: np.ndarray) -> np.ndarray:
    """Convert uncertainty to confidence (1 - uncertainty)."""
    return 1.0 - uncertainty_map
