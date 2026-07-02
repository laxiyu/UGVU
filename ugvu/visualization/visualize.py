"""Visualization tools for UGVU.

Generates:
    - Uncertainty map overlays
    - Side-by-side comparisons (input, prediction, GT, uncertainty)
    - Calibration plots
    - Robustness bar charts
    - Per-class IoU radar charts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ============================================================================
# Uncertainty map visualization
# ============================================================================

def uncertainty_heatmap(
    uncertainty_map: np.ndarray,
    colormap: str = "hot",
    alpha: float = 0.6,
) -> np.ndarray:
    """Convert an uncertainty map to an RGBA heatmap overlay.

    Args:
        uncertainty_map: (H, W) float32 in [0, 1].
        colormap: Matplotlib colormap name ("hot", "viridis", "plasma", "jet").
        alpha: Opacity of the overlay.

    Returns:
        (H, W, 4) uint8 RGBA heatmap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    u_norm = uncertainty_map.copy()
    u_min, u_max = u_norm.min(), u_norm.max()
    if u_max - u_min > 1e-6:
        u_norm = (u_norm - u_min) / (u_max - u_min)

    rgba = cmap(u_norm)  # (H, W, 4)
    rgba[..., 3] *= alpha  # adjust opacity
    return (rgba * 255).astype(np.uint8)


def overlay_uncertainty(
    image: np.ndarray,
    uncertainty_map: np.ndarray,
    colormap: str = "hot",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay uncertainty heatmap on the original image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        uncertainty_map: (H, W) float32 uncertainty.
        colormap: Matplotlib colormap name.
        alpha: Blend strength.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    heatmap = uncertainty_heatmap(uncertainty_map, colormap=colormap, alpha=1.0)
    heatmap_rgb = heatmap[..., :3].astype(np.float32)
    heatmap_alpha = heatmap[..., 3:4].astype(np.float32) / 255.0 * alpha

    image_f = image.astype(np.float32)
    blended = image_f * (1 - heatmap_alpha) + heatmap_rgb * heatmap_alpha
    return blended.clip(0, 255).astype(np.uint8)


# ============================================================================
# Multi-panel visualization
# ============================================================================

def save_comparison_figure(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    uncertainty_map: Optional[np.ndarray],
    output_path: str,
    pred_colormap: Optional[np.ndarray] = None,
    gt_colormap: Optional[np.ndarray] = None,
    title: str = "UGVU Prediction",
) -> None:
    """Save a 4-panel comparison figure: Image | Prediction | Uncertainty | GT.

    Args:
        image: (H, W, 3) uint8 RGB.
        prediction: (H, W) int64 class map.
        ground_truth: (H, W) int64 class map.
        uncertainty_map: (H, W) float32 uncertainty.
        output_path: Path to save PNG.
        pred_colormap: (C, 3) for colorizing prediction.
        gt_colormap: (C, 3) for colorizing GT.
        title: Overall title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4 if uncertainty_map is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Prediction
    if pred_colormap is not None:
        pred_vis = _colorize_mask(prediction, pred_colormap)
    else:
        pred_vis = prediction
    axes[1].imshow(pred_vis)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Uncertainty
    if uncertainty_map is not None:
        im = axes[2].imshow(uncertainty_map, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("Uncertainty (PCU)")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        gt_idx = 3
    else:
        gt_idx = 2

    # Ground Truth
    if gt_colormap is not None:
        gt_vis = _colorize_mask(ground_truth, gt_colormap)
    else:
        gt_vis = ground_truth
    axes[gt_idx].imshow(gt_vis)
    axes[gt_idx].set_title("Ground Truth")
    axes[gt_idx].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _colorize_mask(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Colorize a class-index mask using a colormap."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(colormap.shape[0]):
        rgb[mask == c] = colormap[c]
    return rgb


# ============================================================================
# Robustness bar chart
# ============================================================================

def save_robustness_bar_chart(
    report: Dict,
    output_path: str,
    title: str = "GVU-Robust Scores",
) -> None:
    """Save a bar chart comparing PRS, GRS, MRS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = []
    values = []
    for key, label in [("prs", "PRS"), ("grs", "GRS"), ("mrs", "MRS")]:
        if report.get(key) is not None:
            metrics.append(label)
            values.append(report[key])

    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#3498db", "#e74c3c"][:len(metrics)]
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(1.05, max(values) * 1.2))
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect (1.0)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Per-class IoU radar
# ============================================================================

def save_per_class_iou_chart(
    per_class_iou: Dict[int, float],
    class_names: Optional[List[str]] = None,
    output_path: str = "outputs/per_class_iou.png",
    title: str = "Per-Class IoU",
) -> None:
    """Save a horizontal bar chart of per-class IoU."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(per_class_iou.keys())
    values = [per_class_iou[c] for c in classes]
    labels = [class_names[c] if class_names else f"Class {c}" for c in classes]

    fig, ax = plt.subplots(figsize=(8, max(4, len(classes) * 0.3)))
    colors = plt.cm.viridis(np.array(values) / max(values))

    y_pos = range(len(classes))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("IoU")
    ax.set_title(title)
    ax.axvline(x=np.mean(values), color="red", linestyle="--", label=f"mIoU = {np.mean(values):.3f}")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Save uncertainty as colored PNG
# ============================================================================

def save_uncertainty_map(
    uncertainty_map: np.ndarray,
    output_path: str,
    colormap: str = "hot",
) -> None:
    """Save an uncertainty map as a colored PNG image."""
    rgba = uncertainty_heatmap(uncertainty_map, colormap=colormap, alpha=1.0)
    img = Image.fromarray(rgba[..., :3])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
