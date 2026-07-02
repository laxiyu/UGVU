"""Model Robustness Score (MRS).

Measures how consistent predictions are across different black-box models
given the same image and prompt.

MRS = 1 - (σ / μ)

where σ and μ are computed across models.

Higher MRS → different models agree more → method is model-agnostic.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


def model_robustness_score(metric_values: np.ndarray) -> float:
    """Compute MRS from metrics across models.

    Args:
        metric_values: (M,) float32 metrics (one per model).

    Returns:
        MRS ∈ (-∞, 1].
    """
    mu = metric_values.mean()
    sigma = metric_values.std(ddof=1)
    if mu == 0:
        return 0.0
    return float(1.0 - sigma / abs(mu))


def model_robustness_per_image(
    per_model_metrics: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute MRS aggregated over images.

    Args:
        per_model_metrics: Dict model_name → (N,) float32 metrics per image.

    Returns:
        Dict with mrs_mean, mrs_std, mrs_aggregate, per_model_mrs, per_image_mrs.
    """
    models = list(per_model_metrics.keys())
    M = len(models)
    if M < 2:
        return {"mrs_mean": 1.0, "mrs_std": 0.0, "mrs_aggregate": 1.0,
                "per_image_mrs": np.ones_like(next(iter(per_model_metrics.values())))}

    N = len(next(iter(per_model_metrics.values())))

    # (M, N) matrix
    metric_matrix = np.stack([per_model_metrics[m] for m in models], axis=0)

    # Per-image MRS
    per_image = np.array([model_robustness_score(metric_matrix[:, i]) for i in range(N)])

    # Aggregate: mean metric per model
    mean_per_model = metric_matrix.mean(axis=1)  # (M,)
    mrs_agg = model_robustness_score(mean_per_model)

    return {
        "mrs_mean": float(per_image.mean()),
        "mrs_std": float(per_image.std(ddof=1)),
        "mrs_aggregate": float(mrs_agg),
        "per_image_mrs": per_image,
        "per_model_mean_metric": {m: float(mean_per_model[i]) for i, m in enumerate(models)},
    }


def evaluate_model_robustness(
    images: List[np.ndarray],
    prompt: str,
    model_inference_fns: Dict[str, Callable],
    eval_fn: Callable[[np.ndarray, np.ndarray], float],
    ground_truths: List[np.ndarray],
    verbose: bool = True,
) -> Dict:
    """Run a full model robustness evaluation.

    Args:
        images: List of N input images.
        prompt: A single fixed prompt.
        model_inference_fns: Dict model_name → fn(image, prompt) → prediction.
        eval_fn: function(prediction, ground_truth) → metric.
        ground_truths: List of ground truth arrays.
        verbose: Print progress.

    Returns:
        Full MRS report dict.
    """
    N = len(images)
    per_model = {}

    for model_name, infer_fn in model_inference_fns.items():
        metrics = np.zeros(N)
        for i, (img, gt) in enumerate(zip(images, ground_truths)):
            pred = infer_fn(img, prompt)
            metrics[i] = eval_fn(pred, gt)
        per_model[model_name] = metrics
        if verbose:
            print(f"  Model {model_name}: mean metric = {metrics.mean():.4f}")

    result = model_robustness_per_image(per_model)
    result["per_model_metrics"] = {m: arr.tolist() for m, arr in per_model.items()}
    return result
