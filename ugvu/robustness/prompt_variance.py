"""Prompt Robustness Score (PRS).

Measures how stable predictions are across semantically-equivalent prompts.

PRS = 1 - (σ / μ)

where:
    σ = standard deviation of metric across prompt variants
    μ = mean metric across prompt variants

Higher PRS → more robust to prompt phrasing.
PRS ∈ (-∞, 1]; PRS = 1 means perfect robustness (σ = 0).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


def prompt_robustness_score(
    metric_values: np.ndarray,
) -> float:
    """Compute PRS from a set of metric values across different prompts.

    Args:
        metric_values: (P,) float32 metrics (one per prompt variant).
                       Higher metric = better performance.

    Returns:
        PRS ∈ (-∞, 1].
    """
    mu = metric_values.mean()
    sigma = metric_values.std(ddof=1)
    if mu == 0:
        return 0.0
    return float(1.0 - sigma / abs(mu))


def prompt_robustness_per_image(
    per_prompt_metrics: List[np.ndarray],
    image_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute PRS aggregated over images.

    Args:
        per_prompt_metrics: List of (N,) arrays, one per prompt variant (P variants total).
                            Each array contains metric per image.
        image_indices: Optional subset of images.

    Returns:
        Dict with:
            - prs_mean: Mean PRS over images.
            - prs_std: Std of PRS over images.
            - prs_aggregate: PRS computed from mean metrics (pooled).
            - per_image_prs: (N,) array of per-image PRS.
    """
    P = len(per_prompt_metrics)
    if P < 2:
        return {"prs_mean": 1.0, "prs_std": 0.0, "prs_aggregate": 1.0, "per_image_prs": np.array([1.0])}

    # (P, N) matrix
    metric_matrix = np.stack(per_prompt_metrics, axis=0)  # (P, N)

    if image_indices is not None:
        metric_matrix = metric_matrix[:, image_indices]

    # Per-image PRS
    per_image = []
    for i in range(metric_matrix.shape[1]):
        per_image.append(prompt_robustness_score(metric_matrix[:, i]))
    per_image = np.array(per_image)

    # Aggregate: compute PRS on the mean across images
    mean_per_prompt = metric_matrix.mean(axis=1)  # (P,)
    prs_agg = prompt_robustness_score(mean_per_prompt)

    return {
        "prs_mean": float(per_image.mean()),
        "prs_std": float(per_image.std(ddof=1)),
        "prs_aggregate": float(prs_agg),
        "per_image_prs": per_image,
    }


def evaluate_prompt_robustness(
    images: List[np.ndarray],
    prompts: List[str],
    inference_fn: Callable[[np.ndarray, str], np.ndarray],
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
    ground_truths: List[np.ndarray],
    verbose: bool = True,
) -> Dict:
    """Run a full prompt robustness evaluation.

    Args:
        images: List of (H, W, 3) input images.
        prompts: List of P prompt variants.
        inference_fn: function(image, prompt) → prediction.
        eval_fn: function(prediction, ground_truth) → metric.
        ground_truths: List of ground truth arrays.
        verbose: Print progress.

    Returns:
        Full PRS report dict.
    """
    N = len(images)
    P = len(prompts)

    metric_matrix = np.zeros((P, N))
    for p_idx, prompt in enumerate(prompts):
        for i_idx, (img, gt) in enumerate(zip(images, ground_truths)):
            pred = inference_fn(img, prompt)
            metric_matrix[p_idx, i_idx] = eval_fn(pred, gt)
        if verbose:
            print(f"  Prompt {p_idx+1}/{P}: mean metric = {metric_matrix[p_idx].mean():.4f}")

    per_prompt_metrics = [metric_matrix[p] for p in range(P)]
    result = prompt_robustness_per_image(per_prompt_metrics)

    result["metric_matrix"] = metric_matrix
    result["prompts"] = prompts
    result["mean_per_prompt"] = metric_matrix.mean(axis=1).tolist()
    result["std_per_prompt"] = metric_matrix.std(axis=1).tolist()

    return result
