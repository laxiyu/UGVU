"""Generation Robustness Score (GRS).

Measures how stable predictions are across repeated generations
with the SAME prompt and model (stochasticity of the generator).

GRS = 1 - (σ / μ)

where σ and μ are computed over repeated generations (same prompt, same image).

Higher GRS → generator is more consistent for this task.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


def generation_robustness_score(metric_values: np.ndarray) -> float:
    """Compute GRS from metrics across K repeated generations.

    Args:
        metric_values: (K,) float32 metrics (one per generation repeat).

    Returns:
        GRS ∈ (-∞, 1].
    """
    mu = metric_values.mean()
    sigma = metric_values.std(ddof=1)
    if mu == 0:
        return 0.0
    return float(1.0 - sigma / abs(mu))


def generation_robustness_per_image(
    per_image_metrics: np.ndarray,
) -> Dict[str, float]:
    """Compute GRS across images.

    Args:
        per_image_metrics: (N, K) matrix — N images, K repeats each.

    Returns:
        Dict with grs_mean, grs_std, grs_aggregate, per_image_grs.
    """
    N, K = per_image_metrics.shape
    if K < 2:
        return {"grs_mean": 1.0, "grs_std": 0.0, "grs_aggregate": 1.0, "per_image_grs": np.ones(N)}

    per_image = np.array([generation_robustness_score(per_image_metrics[i]) for i in range(N)])

    # Aggregate: mean metric per repeat
    mean_per_repeat = per_image_metrics.mean(axis=0)  # (K,)
    grs_agg = generation_robustness_score(mean_per_repeat)

    return {
        "grs_mean": float(per_image.mean()),
        "grs_std": float(per_image.std(ddof=1)),
        "grs_aggregate": float(grs_agg),
        "per_image_grs": per_image,
    }


def evaluate_generation_robustness(
    images: List[np.ndarray],
    prompt: str,
    inference_fn: Callable[[np.ndarray, str, int], np.ndarray],
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
    ground_truths: List[np.ndarray],
    k_repeats: int = 20,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict:
    """Run a full generation robustness evaluation.

    Args:
        images: List of N input images.
        prompt: A single fixed prompt.
        inference_fn: function(image, prompt, seed) → prediction.
        eval_fn: function(prediction, ground_truth) → metric.
        ground_truths: List of ground truth arrays.
        k_repeats: Number of generation repeats.
        seeds: Optional per-repeat seeds.
        verbose: Print progress.

    Returns:
        Full GRS report dict.
    """
    N = len(images)
    K = k_repeats

    if seeds is None:
        seeds = list(range(K))

    metric_matrix = np.zeros((N, K))
    for k in range(K):
        seed = seeds[k]
        for i, (img, gt) in enumerate(zip(images, ground_truths)):
            pred = inference_fn(img, prompt, seed)
            metric_matrix[i, k] = eval_fn(pred, gt)
        if verbose:
            print(f"  Repeat {k+1}/{K}: mean metric = {metric_matrix[:, k].mean():.4f}")

    result = generation_robustness_per_image(metric_matrix)
    result["metric_matrix"] = metric_matrix
    result["mean_per_repeat"] = metric_matrix.mean(axis=0).tolist()
    result["std_per_repeat"] = metric_matrix.std(axis=0).tolist()

    return result
