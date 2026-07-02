"""Top-level consensus fusion orchestrator.

Ties together majority vote, UGCF, and CMCF into a single interface.
Selects the appropriate fusion strategy based on configuration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .majority_vote import majority_vote
from .weighted_vote import uncertainty_guided_fusion, uncertainty_guided_fusion_continuous
from .cross_model_vote import (
    cross_model_fusion,
    cross_model_fusion_with_samples,
    cross_model_fusion_continuous,
)


def consensus_fusion(
    mask_stack: np.ndarray,
    uncertainty_map: Optional[np.ndarray] = None,
    method: str = "ugcf",
    num_classes: int = 19,
    ignore_index: int = 255,
    temperature: float = 1.0,
    is_continuous: bool = False,
) -> np.ndarray:
    """Fuse K predictions into a single consensus output.

    Args:
        mask_stack: (K, H, W) int64 (segmentation) or (K, H, W) float32 (depth).
        uncertainty_map: (H, W) or (K, H, W) uncertainty. For "majority", can be None.
                          If (H, W), the same uncertainty is used for all samples
                          (treated as a global per-pixel uncertainty, broadcast).
                          If (K, H, W), per-sample per-pixel uncertainty.
        method: "majority" | "ugcf" | "cmcf".
        num_classes: For categorical tasks.
        ignore_index: Ignore label.
        temperature: Softmax temperature for uncertainty weighting.
        is_continuous: True for depth/continuous tasks.

    Returns:
        (H, W) int64 or float32 fused prediction.
    """
    if method == "majority":
        if is_continuous:
            # Simple mean for depth
            return mask_stack.mean(axis=0).astype(np.float32)
        return majority_vote(mask_stack, ignore_index=ignore_index)

    elif method == "ugcf":
        if uncertainty_map is None:
            raise ValueError("uncertainty_map is required for UGCF method")

        # If uncertainty is (H, W), broadcast to (K, H, W) by treating it as
        # the per-pixel uncertainty of the STACK (not per-sample). In that case,
        # all samples get the same uncertainty weight at each pixel, which is
        # just a regular soft-vote. For per-sample uncertainty, we need (K, H, W).
        if uncertainty_map.ndim == 2:
            per_sample_unc = np.tile(uncertainty_map[None, ...], (mask_stack.shape[0], 1, 1))
        else:
            per_sample_unc = uncertainty_map

        if is_continuous:
            return uncertainty_guided_fusion_continuous(
                mask_stack,
                per_sample_unc,
                temperature=temperature,
            )
        else:
            return uncertainty_guided_fusion(
                mask_stack,
                per_sample_unc,
                num_classes=num_classes,
                ignore_index=ignore_index,
                temperature=temperature,
            )

    elif method == "cmcf":
        # For CMCF, mask_stack should actually be a dict, but if called with
        # a simple stack, fall back to UGCF
        if isinstance(mask_stack, dict):
            return cross_model_fusion(
                mask_stack,
                uncertainty_map if isinstance(uncertainty_map, dict) else {},
                num_classes=num_classes,
                ignore_index=ignore_index,
                temperature=temperature,
            )
        else:
            # Fallback: treat as UGCF
            return consensus_fusion(
                mask_stack,
                uncertainty_map,
                method="ugcf",
                num_classes=num_classes,
                ignore_index=ignore_index,
                temperature=temperature,
                is_continuous=is_continuous,
            )

    else:
        raise ValueError(f"Unknown consensus method: {method}")


def fuse_cross_model(
    model_predictions: Dict[str, np.ndarray],
    model_uncertainties: Dict[str, np.ndarray],
    num_classes: int = 19,
    ignore_index: int = 255,
    temperature: float = 1.0,
    is_continuous: bool = False,
    use_full_samples: bool = False,
) -> np.ndarray:
    """High-level CMCF entry point.

    Args:
        model_predictions: Dict model_name → (H, W) or (K, H, W) predictions.
        model_uncertainties: Dict model_name → (H, W) or (K, H, W) uncertainties.
        num_classes: For categorical tasks.
        ignore_index: Ignore label.
        temperature: Softmax temperature.
        is_continuous: True for depth.
        use_full_samples: If True, use all per-sample predictions.

    Returns:
        (H, W) fused prediction.
    """
    if use_full_samples:
        if is_continuous:
            # Average all
            all_vals = []
            for v in model_predictions.values():
                if v.ndim == 3:
                    all_vals.extend([v[i] for i in range(v.shape[0])])
                else:
                    all_vals.append(v)
            return np.stack(all_vals).mean(axis=0)
        else:
            return cross_model_fusion_with_samples(
                model_predictions,
                model_uncertainties,
                num_classes=num_classes,
                ignore_index=ignore_index,
                temperature=temperature,
            )
    else:
        if is_continuous:
            return cross_model_fusion_continuous(
                model_predictions,
                model_uncertainties,
                temperature=temperature,
            )
        else:
            return cross_model_fusion(
                model_predictions,
                model_uncertainties,
                num_classes=num_classes,
                ignore_index=ignore_index,
                temperature=temperature,
            )
