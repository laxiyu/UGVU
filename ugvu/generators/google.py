"""Google Gemini generator for text-returning multimodal segmentation probes.

This wrapper calls the Gemini generateContent REST endpoint with inline image
bytes, asks for a compact JSON grid, and converts the returned grid into a mask
image compatible with the existing decoders.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

import numpy as np
from PIL import Image

from .base_generator import BaseGenerator
from .qwen import _parse_segmentation_mask

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class GoogleGenerator(BaseGenerator):
    """Google Gemini multimodal generator.

    Environment variables checked in order:
        GOOG_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY
    """

    def __init__(
        self,
        api_endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: str = "",
        model_version: str = "gemini-2.0-flash",
        api_mode: str = "generate_content",
        **kwargs,
    ):
        api_key = (
            api_key
            or os.environ.get("GOOG_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        model_name = kwargs.pop("model_name", "google")
        super().__init__(
            model_name=model_name,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )
        self.api_mode = api_mode
        self.colormap: Optional[np.ndarray] = None

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if not HAS_HTTPX:
            raise ImportError("httpx is required for GoogleGenerator. Install with: pip install httpx")
        if not self.api_key:
            raise RuntimeError("Missing Google API key. Set GOOG_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY.")

        width, height = image.size
        image_b64 = self._encode_image_b64(image)
        grid_h = max(8, height // 32)
        grid_w = max(8, width // 32)
        structured_prompt = (
            f"{prompt}\n\n"
            "Return only a JSON object for semantic segmentation. "
            f"The image size is {width}x{height}. "
            f"Use a reduced grid of about {grid_h} rows and {grid_w} columns. "
            "Format: {\"grid\": [[0,1,...], ...], \"classes\": [\"class0\", ...]}. "
            "Each grid value must be a 0-based class index. Do not use 255, unknown, or text labels."
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": structured_prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": 8192,
                "response_mime_type": "application/json",
            },
        }
        if seed is not None:
            payload["generationConfig"]["seed"] = int(seed)

        endpoint = self.api_endpoint.rstrip("/")
        url = f"{endpoint}/models/{self.model_version}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.is_error:
                raise RuntimeError(f"Google Gemini API {resp.status_code}: {resp.text[:500]}")
            data = resp.json()

        return self._parse_response(data, height, width)

    def _parse_response(self, data: dict, height: int, width: int) -> Image.Image:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in Gemini response. Keys: {list(data.keys())}")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        text = "\n".join(texts)
        if not text:
            raise ValueError(f"No text content in Gemini response. Candidate keys: {list(candidates[0].keys())}")

        mask = _parse_segmentation_mask(text, height, width)
        if self.colormap is not None and mask.max(initial=0) < len(self.colormap):
            rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for class_id in range(len(self.colormap)):
                rgb[mask == class_id] = self.colormap[class_id]
            return Image.fromarray(rgb)
        return Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), mode="L")

    @staticmethod
    def _encode_image_b64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
