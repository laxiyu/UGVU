"""Unified decoder — converts generated RGB images to structured predictions.

The decoder takes raw outputs from black-box generators (which are RGB images
or segmentation-like color maps) and converts them into:
  - Class-index masks (for semantic/referring segmentation)
  - Depth maps (for depth estimation)

Multiple decoding strategies are supported:
  1. Colormap matching (nearest-neighbor in RGB space)
  2. Channel reduction (grayscale → threshold)
  3. Direct pass-through (for already-structured outputs)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ============================================================================
# Base decoder
# ============================================================================

class BaseDecoder:
    """Converts a generated RGB image to a structured prediction array."""

    def decode(self, generated: np.ndarray) -> np.ndarray:
        """Convert a single generated image to a prediction array.

        Args:
            generated: (H, W, 3) uint8 RGB image from the generator.

        Returns:
            (H, W) int64 class-index array, or (H, W) float32 depth array.
        """
        raise NotImplementedError

    def decode_with_uncertainty(self, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode one output and return a per-pixel decoding uncertainty map.

        The default assumes the decoder has no extra reliability signal.
        Specialized decoders can override this with task-specific evidence,
        such as distance from the nearest valid colormap color.
        """
        decoded = self.decode(generated)
        return decoded, np.zeros(decoded.shape, dtype=np.float32)

    def decode_batch(self, generated_stack: np.ndarray) -> np.ndarray:
        """Decode a stack of generated images.

        Args:
            generated_stack: (K, H, W, 3) or (K, H, W) array.

        Returns:
            (K, H, W) decoded predictions.
        """
        results = []
        for i in range(len(generated_stack)):
            results.append(self.decode(generated_stack[i]))
        return np.stack(results, axis=0)

    def decode_batch_with_uncertainty(self, generated_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode a stack and return predictions plus decoding uncertainty."""
        decoded = []
        uncertainties = []
        for i in range(len(generated_stack)):
            pred, unc = self.decode_with_uncertainty(generated_stack[i])
            decoded.append(pred)
            uncertainties.append(unc)
        return np.stack(decoded, axis=0), np.stack(uncertainties, axis=0)


# ============================================================================
# Colormap decoder — for semantic segmentation
# ============================================================================

class ColormapDecoder(BaseDecoder):
    """Decode RGB masks to class indices via nearest-neighbor colormap matching.

    For each pixel in the generated RGB image, find the closest color in the
    dataset's colormap and assign that class index.

    Attributes:
        colormap: (C, 3) uint8 array of class RGB colors.
        num_classes: Number of classes (C).
        ignore_index: Class index to assign when no match is close enough.
        max_distance: Maximum L2 distance in RGB space for a valid match.
                      Pixels exceeding this distance get ignore_index.
    """

    def __init__(
        self,
        colormap: np.ndarray,
        num_classes: int,
        ignore_index: int = 255,
        max_distance: float = 100.0,
    ):
        self.colormap = np.asarray(colormap, dtype=np.float32)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.max_distance = max_distance

        # Build KD-tree for fast nearest-neighbor lookup
        self._tree = cKDTree(self.colormap)

    def decode(self, generated: np.ndarray) -> np.ndarray:
        """Match each pixel to the nearest colormap entry.

        Args:
            generated: (H, W, 3) uint8 RGB image.

        Returns:
            (H, W) int64 class indices.
        """
        h, w = generated.shape[:2]
        if generated.ndim == 3 and generated.shape[2] > 3:
            generated = generated[..., :3]
        pixels = generated.reshape(-1, 3).astype(np.float32)

        distances, indices = self._tree.query(pixels, k=1)

        mask = indices.reshape(h, w).astype(np.int64)
        dist_mask = distances.reshape(h, w)

        # Mark pixels too far from any class color as ignore
        mask[dist_mask > self.max_distance] = self.ignore_index

        return mask

    def decode_with_uncertainty(self, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode RGB masks and expose OOD color distance as uncertainty.

        Pixels that are far from every legal colormap entry are likely invalid
        generator artifacts, so they receive high uncertainty and may be ignored.
        """
        h, w = generated.shape[:2]
        if generated.ndim == 3 and generated.shape[2] > 3:
            generated = generated[..., :3]
        pixels = generated.reshape(-1, 3).astype(np.float32)
        distances, indices = self._tree.query(pixels, k=1)

        mask = indices.reshape(h, w).astype(np.int64)
        dist_mask = distances.reshape(h, w).astype(np.float32)
        invalid = dist_mask > self.max_distance
        mask[invalid] = self.ignore_index

        uncertainty = np.clip(dist_mask / max(self.max_distance, 1e-6), 0.0, 1.0)
        uncertainty[invalid] = 1.0
        return mask, uncertainty.astype(np.float32)


# ============================================================================
# Binary decoder — for referring segmentation
# ============================================================================

class BinaryDecoder(BaseDecoder):
    """Decode RGB output to a binary foreground/background mask.

    Converts the generated image to grayscale and thresholds at 0.5
    (after normalizing to [0, 1]).

    Attributes:
        threshold: Value in [0, 1]; pixels above this are foreground (1).
        invert: If True, darker pixels become foreground.
    """

    def __init__(self, threshold: float = 0.5, invert: bool = False):
        self.threshold = threshold
        self.invert = invert

    def decode(self, generated: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(generated)
        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0
        if self.invert:
            gray = 1.0 - gray
        return (gray > self.threshold).astype(np.int64)

    def decode_with_uncertainty(self, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = self._to_grayscale(generated).astype(np.float32) / 255.0
        if self.invert:
            gray = 1.0 - gray
        mask = (gray > self.threshold).astype(np.int64)
        margin = np.abs(gray - self.threshold)
        uncertainty = np.clip(1.0 - margin / max(self.threshold, 1.0 - self.threshold, 1e-6), 0.0, 1.0)
        return mask, uncertainty.astype(np.float32)

    @staticmethod
    def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB (H, W, 3) to grayscale (H, W)."""
        if rgb.ndim == 2:
            return rgb
        # Standard luminance weights
        return (
            0.2989 * rgb[..., 0].astype(np.float32)
            + 0.5870 * rgb[..., 1].astype(np.float32)
            + 0.1140 * rgb[..., 2].astype(np.float32)
        ).astype(np.uint8)



class ClassIndexDecoder(BaseDecoder):
    """Decode class-index masks stored as grayscale or RGB arrays."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def decode(self, generated: np.ndarray) -> np.ndarray:
        if generated.ndim == 3:
            generated = generated[..., 0]
        mask = np.rint(generated.astype(np.float32)).astype(np.int64)
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        return mask

    def decode_with_uncertainty(self, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if generated.ndim == 3:
            values = generated[..., 0].astype(np.float32)
        else:
            values = generated.astype(np.float32)
        nearest = np.rint(values)
        mask = nearest.astype(np.int64)
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        uncertainty = np.clip(np.abs(values - nearest), 0.0, 1.0).astype(np.float32)
        uncertainty[invalid] = 1.0
        return mask, uncertainty

# ============================================================================
# Depth decoder — for depth estimation
# ============================================================================

class DepthDecoder(BaseDecoder):
    """Decode RGB output to a continuous depth map.

    Strategies:
        - "grayscale": Convert to grayscale, treat intensity as inverse depth.
        - "rgb": Use one channel (default R) as depth proxy.

    Attributes:
        mode: "grayscale" or "rgb".
        channel: Which RGB channel to use when mode="rgb".
        depth_range: (min, max) in meters for rescaling.
        invert: If True, brighter = farther (inverse depth).
    """

    def __init__(
        self,
        mode: str = "grayscale",
        channel: int = 0,
        depth_range: tuple = (0.0, 10.0),
        invert: bool = False,
    ):
        self.mode = mode
        self.channel = channel
        self.depth_range = depth_range
        self.invert = invert

    def decode(self, generated: np.ndarray) -> np.ndarray:
        if self.mode == "grayscale":
            gray = 0.2989 * generated[..., 0].astype(np.float32) \
                 + 0.5870 * generated[..., 1].astype(np.float32) \
                 + 0.1140 * generated[..., 2].astype(np.float32)
        elif self.mode == "rgb":
            gray = generated[..., self.channel].astype(np.float32)
        else:
            raise ValueError(f"Unknown depth decode mode: {self.mode}")

        # Normalize to [0, 1]
        depth_norm = gray / 255.0
        if self.invert:
            depth_norm = 1.0 - depth_norm

        # Rescale to depth range
        dmin, dmax = self.depth_range
        depth = dmin + depth_norm * (dmax - dmin)

        return depth.astype(np.float32)


# ============================================================================
# Factory
# ============================================================================

def build_decoder(
    task: str,
    num_classes: int = 19,
    ignore_index: int = 255,
    colormap: Optional[np.ndarray] = None,
    **kwargs,
) -> BaseDecoder:
    """Build the appropriate decoder for a given task.

    Args:
        task: "semantic_segmentation", "referring_segmentation", or "depth_estimation".
        num_classes: Number of classes for semantic segmentation.
        ignore_index: Ignore label index.
        colormap: (C, 3) colormap for colormap-based decoding.

    Returns:
        Configured BaseDecoder instance.
    """
    if task == "depth_estimation":
        return DepthDecoder(**kwargs)
    elif task == "referring_segmentation":
        return BinaryDecoder(**kwargs)
    elif task == "semantic_segmentation":
        if colormap is not None:
            return ColormapDecoder(
                colormap=colormap,
                num_classes=num_classes,
                ignore_index=ignore_index,
                **kwargs,
            )
        else:
            return ClassIndexDecoder(num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown task: {task}")


