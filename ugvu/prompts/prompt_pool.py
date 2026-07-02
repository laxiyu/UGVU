"""Prompt pool — task-organized templates with perturbation variants for robustness testing.

This module provides:
1. Base prompt templates for each task (semantic segmentation, referring segmentation, depth estimation).
2. Variant generators that produce N semantically-equivalent rewrites.
3. A PromptPool class that manages prompt selection and perturbation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ============================================================================
# Base templates
# ============================================================================

SEMANTIC_SEGMENTATION_TEMPLATES = [
    "Segment all objects in this image and assign each pixel a class label.",
    "Perform semantic segmentation on this image. Label every pixel with its object category.",
    "Generate a dense semantic segmentation mask for this scene. Each pixel must be classified.",
    "Parse the scene: output a per-pixel class map identifying every object category present.",
    "Annotate this image with pixel-level semantic labels for all visible objects.",
    "Produce a full semantic segmentation map. Assign a class ID to each and every pixel.",
    "Label each pixel in this image with the semantic class it belongs to.",
    "Create a pixel-wise classification of the scene — every object, every pixel labeled.",
    "Do dense semantic labeling: give me a mask where each pixel gets its object class.",
    "Generate a complete segmentation mask for the image, classifying all pixels into semantic categories.",
]

REFERRING_SEGMENTATION_TEMPLATES = [
    "Segment the {object} in this image and return a binary mask.",
    "Find and segment the {object}. Output a binary mask where the object is white and background is black.",
    "Locate the {object} in this image and produce a segmentation mask for it.",
    "Given the referring expression '{object}', segment the corresponding region.",
    "Generate a mask highlighting the {object} in the scene.",
    "Identify and segment the {object}. Return a pixel mask for it.",
    "Produce a binary segmentation mask for the {object}.",
    "Which pixels belong to the {object}? Output a binary mask.",
    "Isolate the {object}: generate a mask covering only that object.",
    "Create a pixel mask for the region described as '{object}'.",
]

DEPTH_ESTIMATION_TEMPLATES = [
    "Estimate the depth of every pixel in this image. Output a depth map (brighter = closer).",
    "Generate a dense depth map for this scene. Near objects should be bright, far objects dark.",
    "Produce a per-pixel depth estimation. Brighter pixels represent points closer to the camera.",
    "Infer the 3D structure: output a monocular depth map for this image.",
    "Estimate metric depth for each pixel and return a depth image.",
    "Predict the distance from the camera to each pixel and visualize as a depth map.",
    "Generate an inverse depth map — brighter means nearer to the viewpoint.",
    "Create a depth estimation: assign a depth value to each pixel (bright = near, dark = far).",
    "Perform monocular depth estimation and output a dense depth prediction.",
    "Compute a depth map for this image. Nearer surfaces should appear brighter.",
]


# ============================================================================
# Variant generators (for robustness testing)
# ============================================================================

VERB_SYNONYMS = {
    "segment":   ["segment", "detect", "identify", "extract", "delineate", "outline", "partition"],
    "generate":  ["generate", "produce", "create", "output", "return", "build", "render"],
    "label":     ["label", "annotate", "tag", "classify", "mark", "assign"],
    "estimate":  ["estimate", "predict", "infer", "compute", "determine", "calculate", "approximate"],
    "find":      ["find", "locate", "identify", "detect", "spot", "discover"],
    "parse":     ["parse", "analyze", "interpret", "understand", "decompose", "break down"],
    "mask":      ["mask", "segmentation mask", "pixel mask", "binary map", "region map"],
    "highlight": ["highlight", "mark", "indicate", "show", "visualize", "delineate"],
}

NOISE_PREFIXES = [
    "",
    "Please ",
    "I need you to ",
    "Your task is to ",
    "Can you ",
    "You must ",
    "I want you to ",
    "Kindly ",
]

NOISE_SUFFIXES = [
    "",
    ".",
    ". Do it carefully.",
    ". Be precise.",
    ". Ensure accuracy.",
    ". Provide only the result.",
    ". Return just the mask.",
    ". No text, only the output image.",
]


def _apply_verb_synonyms(text: str) -> str:
    """Replace verbs with random synonyms."""
    words = text.split()
    for i, w in enumerate(words):
        low = w.lower().rstrip(".,;:!?")
        if low in VERB_SYNONYMS:
            synonym = random.choice(VERB_SYNONYMS[low])
            # Preserve capitalization
            if w[0].isupper():
                synonym = synonym[0].upper() + synonym[1:]
            suffix = w[len(low):]  # punctuation
            words[i] = synonym + suffix
    return " ".join(words)


def generate_prompt_variants(
    base_templates: List[str],
    n: int,
    seed: int = 42,
    add_noise: bool = True,
    apply_synonyms: bool = True,
) -> List[str]:
    """Generate N prompt variants from a set of base templates.

    Variants are created by:
    1. Picking a random base template (with replacement)
    2. Optionally replacing verbs with synonyms
    3. Optionally adding noise prefixes/suffixes

    Args:
        base_templates: List of base prompt strings.
        n: Number of variants to generate.
        seed: Random seed for reproducibility.
        add_noise: Whether to add noise prefixes/suffixes.
        apply_synonyms: Whether to apply verb synonym replacement.

    Returns:
        List of N prompt variant strings.
    """
    rng = random.Random(seed)
    variants = []
    for i in range(n):
        # Deterministic seed per variant
        rng_v = random.Random(seed + i * 137 + 1)
        template = rng_v.choice(base_templates)
        if apply_synonyms:
            template = _apply_verb_synonyms(template)
        if add_noise:
            prefix = rng_v.choice(NOISE_PREFIXES)
            suffix = rng_v.choice(NOISE_SUFFIXES)
            template = f"{prefix}{template}{suffix}"
        variants.append(template.strip())
    return variants


# ============================================================================
# PromptPool
# ============================================================================

@dataclass
class PromptPool:
    """Manages prompt templates and variants for all tasks.

    Attributes:
        task: The task name ("semantic_segmentation", "referring_segmentation", "depth_estimation").
        base_templates: The list of base prompt templates for the task.
        default_prompt: The default prompt to use for single-generation runs.
        variants: Pre-generated pool of prompt variants (lazy).
    """

    task: str
    base_templates: List[str] = field(default_factory=list)
    default_prompt: str = ""
    variants: List[str] = field(default_factory=list)
    _variant_pointer: int = 0

    def __post_init__(self):
        if not self.base_templates:
            self.base_templates = self._load_templates()
        if not self.default_prompt:
            self.default_prompt = self.base_templates[0]

    def _load_templates(self) -> List[str]:
        mapping = {
            "semantic_segmentation": SEMANTIC_SEGMENTATION_TEMPLATES,
            "referring_segmentation": REFERRING_SEGMENTATION_TEMPLATES,
            "depth_estimation": DEPTH_ESTIMATION_TEMPLATES,
        }
        return mapping.get(self.task, SEMANTIC_SEGMENTATION_TEMPLATES)

    def generate_variants(self, n: int, seed: int = 42, **kwargs) -> List[str]:
        """Generate (or regenerate) N prompt variants."""
        self.variants = generate_prompt_variants(self.base_templates, n=n, seed=seed, **kwargs)
        self._variant_pointer = 0
        return self.variants

    def next_variant(self) -> str:
        """Return the next variant in round-robin order. Wraps around."""
        if not self.variants:
            self.generate_variants(n=50)
        variant = self.variants[self._variant_pointer]
        self._variant_pointer = (self._variant_pointer + 1) % len(self.variants)
        return variant

    def get_prompt(self, variant_index: Optional[int] = None) -> str:
        """Get a specific prompt by variant index, or the default."""
        if variant_index is not None and self.variants:
            return self.variants[variant_index % len(self.variants)]
        return self.default_prompt

    def fill_referring(self, expression: str, variant_index: Optional[int] = None) -> str:
        """Fill a referring expression template with the given object description."""
        template = self.get_prompt(variant_index)
        return template.format(object=expression)


# ============================================================================
# Factory
# ============================================================================

PROMPT_POOL_CACHE: Dict[str, PromptPool] = {}


def get_prompt_pool(task: str) -> PromptPool:
    """Get or create a PromptPool for the given task."""
    if task not in PROMPT_POOL_CACHE:
        PROMPT_POOL_CACHE[task] = PromptPool(task=task)
    return PROMPT_POOL_CACHE[task]
