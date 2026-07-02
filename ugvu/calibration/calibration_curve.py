"""Calibration curve plotting — matplotlib-based visualization.

Generates:
    - Reliability diagrams (confidence vs. accuracy)
    - Uncertainty vs. Error scatter plots
    - ECE bar charts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


def plot_reliability_diagram(
    bin_confidences: list,
    bin_accuracies: list,
    bin_counts: list,
    ece: float,
    output_path: Optional[str] = None,
    title: str = "Reliability Diagram",
    figsize: tuple = (7, 6),
) -> "plt.Figure":
    """Plot a reliability diagram (calibration curve).

    Args:
        bin_confidences: Mean confidence per bin.
        bin_accuracies: Mean accuracy per bin.
        bin_counts: Sample count per bin.
        ece: Expected Calibration Error value.
        output_path: Path to save the figure.
        title: Plot title.
        figsize: Figure size (inches).

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_conf = np.array(bin_confidences)
    bin_acc = np.array(bin_accuracies)
    bin_cnt = np.array(bin_counts)

    fig, ax = plt.subplots(figsize=figsize)

    # Bar widths proportional to bin count
    total = bin_cnt.sum()
    widths = bin_cnt / total * 0.8 if total > 0 else np.full_like(bin_cnt, 0.08)

    # Bar chart for each bin
    colors = plt.cm.RdYlGn(bin_acc / max(bin_acc.max(), 0.01))
    bars = ax.bar(
        bin_conf,
        bin_acc,
        width=widths,
        alpha=0.7,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    # Gap annotations for large deviations
    for i, (c, a) in enumerate(zip(bin_conf, bin_acc)):
        gap = abs(c - a)
        if gap > 0.05 and bin_cnt[i] > 0:
            ax.annotate(
                f"{gap:.2f}",
                (c, (c + a) / 2),
                fontsize=7,
                ha="center",
                color="red",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (1 - Uncertainty)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_uncertainty_error_scatter(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    max_points: int = 5000,
    output_path: Optional[str] = None,
    title: str = "Uncertainty vs. Error",
    figsize: tuple = (8, 5),
) -> "plt.Figure":
    """Plot uncertainty vs. error as a scatter/density plot.

    X-axis: uncertainty percentile (0=low, 1=high).
    Y-axis: error rate per uncertainty percentile.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u_flat = uncertainty_map.ravel()
    e_flat = error_map.ravel().astype(np.float32)
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    u = u_flat[valid]
    e = e_flat[valid]

    if len(u) > max_points:
        idx = np.random.RandomState(42).choice(len(u), max_points, replace=False)
        u = u[idx]
        e = e[idx]

    # Compute error rate per uncertainty percentile
    percentiles = np.linspace(0, 100, 21)
    pct_edges = np.percentile(u, percentiles)
    bin_error_rates = []
    bin_centers = []
    for i in range(len(pct_edges) - 1):
        mask = (u >= pct_edges[i]) & (u < pct_edges[i + 1])
        if mask.sum() > 0:
            bin_error_rates.append(e[mask].mean())
            bin_centers.append((pct_edges[i] + pct_edges[i + 1]) / 2)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: scatter
    axes[0].scatter(u, e, alpha=0.3, s=1, c="blue", edgecolors="none")
    axes[0].set_xlabel("Uncertainty", fontsize=11)
    axes[0].set_ylabel("Error (1=wrong, 0=correct)", fontsize=11)
    axes[0].set_title("Per-Pixel Uncertainty vs Error", fontsize=12)

    # Right: binned error rate
    axes[1].plot(bin_centers, bin_error_rates, "o-", color="red", linewidth=2, markersize=5)
    axes[1].set_xlabel("Uncertainty", fontsize=11)
    axes[1].set_ylabel("Error Rate", fontsize=11)
    axes[1].set_title("Binned Error Rate by Uncertainty", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_calibration_summary(
    ece_result: dict,
    correlation_result: dict,
    output_path: Optional[str] = None,
    title: str = "Calibration Summary",
) -> "plt.Figure":
    """Plot a comprehensive calibration summary figure.

    Combines reliability diagram, ECE bar, and correlation stats.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))

    # --- Reliability diagram ---
    ax1 = fig.add_subplot(1, 3, 1)
    bin_conf = np.array(ece_result["bin_confidences"])
    bin_acc = np.array(ece_result["bin_accuracies"])
    bin_cnt = np.array(ece_result["bin_counts"])
    non_empty = bin_cnt > 0
    widths = bin_cnt[non_empty] / bin_cnt.sum() * 0.8 if bin_cnt.sum() > 0 else np.full_like(bin_cnt[non_empty], 0.08)
    ax1.bar(bin_conf[non_empty], bin_acc[non_empty], width=widths, alpha=0.7, edgecolor="black")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram\nECE={ece_result['ece']:.4f}  MCE={ece_result['mce']:.4f}")
    ax1.grid(True, alpha=0.3)

    # --- Bar chart of bin gaps ---
    ax2 = fig.add_subplot(1, 3, 2)
    gaps = np.abs(bin_conf[non_empty] - bin_acc[non_empty])
    colors = plt.cm.RdYlGn(1 - gaps / max(gaps.max(), 0.01))
    ax2.bar(range(len(gaps)), gaps, color=colors, edgecolor="black")
    ax2.axhline(y=ece_result["ece"], color="red", linestyle="--", label=f"ECE = {ece_result['ece']:.4f}")
    ax2.set_xlabel("Bin")
    ax2.set_ylabel("|Conf - Acc|")
    ax2.set_title("Per-Bin Calibration Gap")
    ax2.legend(fontsize=9)

    # --- Correlation summary ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    metrics_text = (
        f"Spearman ρ:  {correlation_result.get('spearman_r', 0):.4f}\n"
        f"  p-value:    {correlation_result.get('spearman_p', 1):.2e}\n\n"
        f"Pearson r:    {correlation_result.get('pearson_r', 0):.4f}\n"
        f"  p-value:    {correlation_result.get('pearson_p', 1):.2e}\n\n"
        f"AUROC:        {correlation_result.get('auroc', 0.5):.4f}"
    )
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax3.set_title("Correlation Analysis")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
