"""Base generator — abstract interface for black-box image generation APIs.

All model-specific generators inherit from this class and implement:
    generate(image, prompt) -> PIL.Image
"""

from __future__ import annotations

import time
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for image generation API wrappers.

    Each concrete generator wraps a specific black-box model (Doubao, Qwen, Flux, etc.)
    and exposes a uniform `generate(image, prompt) -> Image` interface.

    Attributes:
        model_name: Human-readable model identifier.
        config: ModelConfig dataclass with API endpoint, key, timeout, etc.
        cache_dir: Directory for caching generated outputs (optional).
    """

    def __init__(
        self,
        model_name: str,
        api_endpoint: str = "",
        api_key: str = "",
        model_version: str = "latest",
        timeout_sec: int = 60,
        max_retries: int = 3,
        image_size: tuple = (1024, 1024),
        temperature: float = 1.0,
        cache_dir: Optional[str] = None,
        **extra_params,
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_version = model_version
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.image_size = image_size
        self.temperature = temperature
        self.extra_params = extra_params
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._call_count = 0
        self._total_time = 0.0

    # ---- Public API ----

    def generate(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate an output image given an input image and text prompt.

        Checks cache first, then calls the remote API with retries.
        """
        cache_key = self._cache_key(image, prompt, seed)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug(f"[{self.model_name}] Cache hit: {cache_key[:16]}...")
            return cached

        t0 = time.perf_counter()
        result = self._call_with_retries(image, prompt, seed)
        elapsed = time.perf_counter() - t0
        self._call_count += 1
        self._total_time += elapsed
        logger.info(f"[{self.model_name}] Generation #{self._call_count} in {elapsed:.1f}s")

        self._save_cache(cache_key, result)
        return result

    def generate_batch(
        self,
        image: Image.Image,
        prompts: list[str],
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        """Generate multiple outputs for the same image using different prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            s = (seed + i) if seed is not None else None
            results.append(self.generate(image, prompt, seed=s))
        return results

    # ---- Subclass interface ----

    @abstractmethod
    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Make a single API call. Must be implemented by subclasses."""
        ...

    def _call_with_retries(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Call the API with retry logic for transient failures."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._call_api(image, prompt, seed)
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"[{self.model_name}] Attempt {attempt+1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        raise RuntimeError(
            f"[{self.model_name}] All {self.max_retries} attempts failed. Last error: {last_error}"
        )

    # ---- Caching ----

    def _cache_key(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> str:
        """Create a deterministic cache key from inputs."""
        hasher = hashlib.sha256()
        # Hash image bytes
        img_bytes = image.tobytes()
        hasher.update(img_bytes)
        # Hash prompt
        hasher.update(prompt.encode("utf-8"))
        # Hash seed
        hasher.update(str(seed).encode("utf-8"))
        # Hash model
        hasher.update(self.model_name.encode("utf-8"))
        return hasher.hexdigest()

    def _load_cache(self, key: str) -> Optional[Image.Image]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.png"
        if path.exists():
            return Image.open(path)
        return None

    def _save_cache(self, key: str, img: Image.Image) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.png"
        img.save(path)

    # ---- Stats ----

    @property
    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "calls": self._call_count,
            "total_time_sec": round(self._total_time, 1),
            "avg_time_sec": round(self._total_time / max(self._call_count, 1), 2),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, calls={self._call_count})"


# ============================================================================
# Mock generator for offline development & testing
# ============================================================================

class MockGenerator(BaseGenerator):
    """A mock generator that returns perturbed copies of the input image.

    Useful for offline development, unit tests, and CI without API credentials.
    Simulates the black-box generation process by:
        - Adding Gaussian noise to simulate generation stochasticity
        - Optionally color-quantizing to class colors
        - Respecting seed for reproducibility

    Attributes:
        noise_std: Standard deviation of additive Gaussian noise.
        quantize_colors: If True and a colormap is provided, quantize output to those colors.
        colormap: (N, 3) array of RGB colors for class quantization.
    """

    def __init__(
        self,
        model_name: str = "mock",
        noise_std: float = 0.05,
        quantize_colors: bool = False,
        colormap: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.noise_std = noise_std
        self.quantize_colors = quantize_colors
        self.colormap = colormap

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        rng = np.random.RandomState(seed)
        arr = np.array(image).astype(np.float32) / 255.0

        # Add Gaussian noise
        noise = rng.randn(*arr.shape).astype(np.float32) * self.noise_std
        arr = np.clip(arr + noise, 0.0, 1.0)

        # Quantize to colormap if requested
        if self.quantize_colors and self.colormap is not None:
            h, w, _ = arr.shape
            cmap = self.colormap.astype(np.float32) / 255.0
            # Nearest neighbor per pixel
            flat = arr.reshape(-1, 3)
            dists = np.sum((flat[:, None, :] - cmap[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(dists, axis=1)
            arr = cmap[nearest].reshape(h, w, 3)

        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)
