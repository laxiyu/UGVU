"""K-shot sampler — orchestrates multi-sample generation across prompts and models.

The Sampler is the entry point for the Generation Layer:
    Image + Prompt Pool → K samples (across models) → SampleCollection
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# SampleCollection — container for K generated masks
# ============================================================================

@dataclass
class SampleCollection:
    """Holds K generated samples for a single image.

    Attributes:
        image_id: Unique identifier for the source image.
        source_image: The original PIL Image.
        masks: List of K generated masks (PIL Image or numpy array).
        prompts: The prompt used for each sample.
        model_names: Which model produced each sample.
        seeds: Per-sample random seed.
        metadata: Arbitrary extra metadata.
    """

    image_id: str
    source_image: Image.Image
    masks: List[np.ndarray] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    model_names: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.masks)

    @property
    def mask_stack(self) -> np.ndarray:
        """Return masks stacked as (K, H, W) or (K, H, W, C).

        Remote APIs sometimes return slightly different mask sizes across
        samples. Normalize to the source image resolution before stacking.
        """
        if not self.masks:
            return np.array([])
        target_size = self.source_image.size
        normalized = []
        for mask in self.masks:
            arr = np.asarray(mask)
            if arr.shape[:2] != (target_size[1], target_size[0]):
                pil = Image.fromarray(arr.astype(np.uint8))
                arr = np.asarray(pil.resize(target_size, resample=Image.NEAREST))
            normalized.append(arr)
        return np.stack(normalized, axis=0)

    def add_sample(self, mask: np.ndarray, prompt: str = "", model: str = "", seed: int = 0):
        self.masks.append(mask)
        self.prompts.append(prompt)
        self.model_names.append(model)
        self.seeds.append(seed)

    def __repr__(self) -> str:
        return f"SampleCollection(id={self.image_id!r}, k={self.k})"


# ============================================================================
# Sampler
# ============================================================================

class Sampler:
    """Orchestrates K-shot sampling from one or more black-box generators.

    For a given image:
        1. Select K prompts from the prompt pool (or use provided prompts).
        2. Distribute prompts across available models.
        3. Call each (model, prompt) pair, collecting results.
        4. Return a SampleCollection.

    Supports both sequential and parallel (threaded) execution.

    Attributes:
        generators: Dict mapping model name → BaseGenerator instance.
        k_samples: Number of total samples to generate per image.
        parallel: Whether to run generator calls in parallel.
        max_concurrent: Max concurrent API calls (for parallel mode).
        seed: Base random seed.
    """

    def __init__(
        self,
        generators: Dict[str, BaseGenerator],
        k_samples: int = 5,
        parallel: bool = True,
        max_concurrent: int = 4,
        seed: int = 42,
    ):
        self.generators = generators
        self.k_samples = k_samples
        self.parallel = parallel
        self.max_concurrent = max_concurrent
        self.seed = seed

    # ---- Core API ----

    def sample(
        self,
        image: Image.Image,
        prompts: List[str],
        models: Optional[List[str]] = None,
        image_id: str = "unknown",
        seed_offset: int = 0,
    ) -> SampleCollection:
        """Generate K samples for a single image.

        Args:
            image: Input PIL Image.
            prompts: List of K (or more) text prompts.
            models: List of model names to distribute across. If None, uses all available.
            image_id: Identifier for the image.

        Returns:
            SampleCollection with K masks.
        """
        if models is None:
            models = list(self.generators.keys())
        if not models:
            raise ValueError("No models available for sampling.")

        # Ensure we have exactly K prompts
        k = self.k_samples
        if len(prompts) < k:
            # Cycle prompts
            prompts = (prompts * ((k // len(prompts)) + 1))[:k]
        prompts = prompts[:k]

        # Assign models round-robin
        assignments = [(models[i % len(models)], prompts[i], self.seed + seed_offset + i) for i in range(k)]

        collection = SampleCollection(image_id=image_id, source_image=image)

        if self.parallel and len(assignments) > 1:
            collection = self._sample_parallel(image, assignments, collection)
        else:
            collection = self._sample_sequential(image, assignments, collection)

        return collection

    def sample_multi_image(
        self,
        images: List[Image.Image],
        prompts: List[str],
        models: Optional[List[str]] = None,
        image_ids: Optional[List[str]] = None,
    ) -> List[SampleCollection]:
        """Generate samples for multiple images."""
        if image_ids is None:
            image_ids = [f"img_{i:04d}" for i in range(len(images))]

        results = []
        for img, img_id in zip(images, image_ids):
            results.append(self.sample(img, prompts=prompts, models=models, image_id=img_id))
        return results

    # ---- Internal ----

    def _sample_sequential(
        self,
        image: Image.Image,
        assignments: List[tuple],
        collection: SampleCollection,
    ) -> SampleCollection:
        for model_name, prompt, seed in assignments:
            gen = self.generators[model_name]
            result_img = gen.generate(image, prompt, seed=seed)
            mask = np.array(result_img)
            collection.add_sample(mask, prompt=prompt, model=model_name, seed=seed)
        return collection

    def _sample_parallel(
        self,
        image: Image.Image,
        assignments: List[tuple],
        collection: SampleCollection,
    ) -> SampleCollection:
        results: List[Optional[tuple]] = [None] * len(assignments)

        def _task(idx: int, model_name: str, prompt: str, seed: int):
            gen = self.generators[model_name]
            result_img = gen.generate(image, prompt, seed=seed)
            mask = np.array(result_img)
            return idx, mask, prompt, model_name, seed

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {}
            for i, (model_name, prompt, seed) in enumerate(assignments):
                fut = executor.submit(_task, i, model_name, prompt, seed)
                futures[fut] = i

            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, mask, prompt, model_name, seed = fut.result()
                    results[idx] = (mask, prompt, model_name, seed)
                except Exception as e:
                    logger.error(f"Sample {futures[fut]} failed: {e}")

        for res in results:
            if res is not None:
                mask, prompt, model_name, seed = res
                collection.add_sample(mask, prompt=prompt, model=model_name, seed=seed)
            else:
                # Fill with a placeholder to keep K constant
                h, w = np.array(image).shape[:2]
                collection.add_sample(
                    np.zeros((h, w), dtype=np.uint8),
                    prompt="(failed)",
                    model="unknown",
                    seed=-1,
                )

        return collection

    # ---- Stats ----

    @property
    def stats(self) -> dict:
        return {name: gen.stats for name, gen in self.generators.items()}


# ============================================================================
# Factory
# ============================================================================

def build_sampler(
    model_configs: dict,
    k_samples: int = 5,
    parallel: bool = True,
    max_concurrent: int = 4,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Sampler:
    """Build a Sampler from a model config dictionary.

    Args:
        model_configs: Dict of model_name → {class, api_endpoint, api_key, ...}.
        k_samples: Default number of samples per image.
        parallel: Enable parallel generation.
        max_concurrent: Max concurrent threads.
        seed: Base seed.
        cache_dir: Shared cache directory.

    Returns:
        Configured Sampler instance.
    """
    from .doubao import DoubaoGenerator
    from .qwen import QwenGenerator
    from .flux import FluxGenerator
    from .google import GoogleGenerator
    from .rightcode import RightCodeGenerator
    from .base_generator import MockGenerator

    MODEL_CLASSES = {
        "doubao": DoubaoGenerator,
        "seedream": DoubaoGenerator,  # 同属 Ark 平台，通过 api_mode 区分
        "qwen": QwenGenerator,
        "flux": FluxGenerator,
        "google": GoogleGenerator,
        "gemini": GoogleGenerator,
        "rightcode": RightCodeGenerator,
        "gpt-image-2-vip": RightCodeGenerator,
        "gpt-image-2": RightCodeGenerator,
        "nano-banana": RightCodeGenerator,
        "nano-banana-2": RightCodeGenerator,
        "nano-banana-pro": RightCodeGenerator,
        "mock": MockGenerator,
    }

    generators = {}
    for name, cfg in model_configs.items():
        cls_name = cfg.pop("class", name)
        cls = MODEL_CLASSES.get(cls_name, MockGenerator)
        generators[name] = cls(
            model_name=name,
            cache_dir=str(Path(cache_dir) / name) if cache_dir else None,
            **cfg,
        )

    return Sampler(
        generators=generators,
        k_samples=k_samples,
        parallel=parallel,
        max_concurrent=max_concurrent,
        seed=seed,
    )



