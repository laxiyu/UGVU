"""Cross-Model Consensus Fusion (CMCF).

CMCF fuses the consensus prediction from multiple black-box models using
spatially varying uncertainty weights. This is deliberately pixel-wise: a model
can be trusted in one region and down-weighted in another.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _softmax_over_models(score_stack: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax over the first axis."""
    logits = score_stack / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-8)


def cross_model_pixel_weights(
    model_uncertainties: Dict[str, np.ndarray],
    temperature: float = 1.0,
    per_model_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Return per-pixel model weights derived from inverse uncertainty.

    Args:
        model_uncertainties: Dict model name -> (H, W) uncertainty map.
        temperature: Lower values make model selection sharper.
        per_model_weights: Optional scalar priors applied before softmax.

    Returns:
        Dict model name -> (H, W) normalized weights that sum to one per pixel.
    """
    models = list(model_uncertainties.keys())
    if not models:
        return {}

    score_maps = []
    for model in models:
        unc = model_uncertainties[model].astype(np.float32)
        if unc.ndim == 3:
            unc = unc.mean(axis=0)
        prior = 1.0 if per_model_weights is None else float(per_model_weights.get(model, 1.0))
        score_maps.append((-unc + np.log(max(prior, 1e-8))).astype(np.float32))

    weights = _softmax_over_models(np.stack(score_maps, axis=0), temperature)
    return {model: weights[i].astype(np.float32) for i, model in enumerate(models)}


def cross_model_fusion(
    model_masks: Dict[str, np.ndarray],
    model_uncertainties: Dict[str, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    temperature: float = 1.0,
    per_model_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Fuse predictions from multiple models with pixel-wise uncertainty weights.

    Formula:
        model_weight_m(p) = softmax_m(-uncertainty_m(p) / tau)
        final_class(p) = argmax_c sum_m model_weight_m(p) * 1[mask_m(p) == c]
    """
    models = list(model_masks.keys())
    H, W = next(iter(model_masks.values())).shape
    norm_weights = cross_model_pixel_weights(
        {m: model_uncertainties[m] for m in models},
        temperature=temperature,
        per_model_weights=per_model_weights,
    )

    vote = np.zeros((num_classes, H, W), dtype=np.float32)
    for model in models:
        mask = model_masks[model]
        weights = norm_weights[model]
        valid = mask != ignore_index
        for cls in range(num_classes):
            match = valid & (mask == cls)
            vote[cls][match] += weights[match]

    result = vote.argmax(axis=0).astype(np.int64)
    total = vote.sum(axis=0)
    result[total == 0] = ignore_index
    return result


def cross_model_fusion_with_samples(
    all_model_masks: Dict[str, np.ndarray],
    all_model_uncertainties: Dict[str, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    temperature: float = 1.0,
) -> np.ndarray:
    """CMCF with full per-sample stacks from multiple models.

    This treats all samples from all models as a single pool with per-sample
    uncertainty weighting (a combined UGCF + CMCF).
    """
    all_masks_list = []
    all_unc_list = []

    for model_name, masks in all_model_masks.items():
        uncs = all_model_uncertainties[model_name]
        for idx in range(masks.shape[0]):
            all_masks_list.append(masks[idx])
            all_unc_list.append(uncs[idx])

    combined_masks = np.stack(all_masks_list, axis=0)
    combined_uncs = np.stack(all_unc_list, axis=0)

    from .weighted_vote import uncertainty_guided_fusion

    return uncertainty_guided_fusion(
        combined_masks,
        combined_uncs,
        num_classes=num_classes,
        ignore_index=ignore_index,
        temperature=temperature,
    )


def cross_model_fusion_continuous(
    model_values: Dict[str, np.ndarray],
    model_uncertainties: Dict[str, np.ndarray],
    temperature: float = 1.0,
    per_model_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """CMCF for continuous predictions such as depth."""
    models = list(model_values.keys())
    weights = cross_model_pixel_weights(
        {m: model_uncertainties[m] for m in models},
        temperature=temperature,
        per_model_weights=per_model_weights,
    )

    result = np.zeros_like(next(iter(model_values.values())), dtype=np.float32)
    for model in models:
        result += weights[model] * model_values[model]
    return result
