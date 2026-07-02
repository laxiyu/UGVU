"""Correlation analysis between uncertainty and actual error."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def spearman_correlation(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
) -> float:
    """Compute Spearman rank correlation between uncertainty and error."""
    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.float32)
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    if valid.sum() < 3:
        return 0.0
    u = u_flat[valid]
    e = e_flat[valid]
    if len(np.unique(u)) < 2 or len(np.unique(e)) < 2:
        return 0.0
    return _pearson_1d(_rankdata_average(u), _rankdata_average(e))


def pearson_correlation(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
) -> float:
    """Compute Pearson correlation between uncertainty and error."""
    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.float32)
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    if valid.sum() < 3:
        return 0.0
    return _pearson_1d(u_flat[valid], e_flat[valid])


def uncertainty_error_auroc(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
) -> float:
    """Compute AUROC with uncertainty as score and error as positive class."""
    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.int32)
    valid = np.isfinite(u_flat)
    scores = u_flat[valid]
    labels = e_flat[valid]

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _rankdata_average(scores)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def correlation_report(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
) -> Dict[str, float]:
    """Generate a full correlation report."""
    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.float32)
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    u = u_flat[valid]
    e = e_flat[valid]

    if len(u) < 3 or len(np.unique(u)) < 2 or len(np.unique(e)) < 2:
        return {"spearman_r": 0.0, "spearman_p": 1.0, "pearson_r": 0.0, "pearson_p": 1.0, "auroc": 0.5}

    return {
        "spearman_r": spearman_correlation(uncertainty_map, error_map),
        "spearman_p": 1.0,
        "pearson_r": pearson_correlation(uncertainty_map, error_map),
        "pearson_p": 1.0,
        "auroc": uncertainty_error_auroc(uncertainty_map, error_map),
    }
