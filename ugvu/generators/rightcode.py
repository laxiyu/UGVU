"""Right Code image generator wrapper.

This adapter targets the Right Code draw chat-completions endpoint documented at:
https://docs.right.codes/docs/rc_extension/draw/chat-completions.html

The endpoint is OpenAI-style:
    POST https://www.right.codes/draw/v1/chat/completions
    Authorization: Bearer ${IMAGE_API_KEY}

It is used here for image-to-image segmentation probes with the `image-2` and
`nanobanana` model identifiers.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Optional

import numpy as np
from PIL import Image

from .base_generator import BaseGenerator
from .doubao import _decode_data_url, _download_image

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class RightCodeGenerator(BaseGenerator):
    """Right Code draw API generator.

    Environment variables checked in order:
        IMAGE_API_KEY, RIGHT_CODE_API_KEY, RIGHTCODES_API_KEY
    """

    def __init__(
        self,
        api_endpoint: str = "https://www.right.codes/draw/v1",
        api_key: str = "",
        model_version: str = "image-2",
        api_mode: str = "chat_completions",
        **kwargs,
    ):
        api_key = (
            api_key
            or os.environ.get("IMAGE_API_KEY", "")
            or os.environ.get("RIGHT_CODE_API_KEY", "")
            or os.environ.get("RIGHTCODES_API_KEY", "")
        )
        model_name = kwargs.pop("model_name", model_version)
        super().__init__(
            model_name=model_name,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )
        self.api_mode = api_mode

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if not HAS_HTTPX:
            raise ImportError("httpx is required for RightCodeGenerator. Install with: pip install httpx")
        if not self.api_key:
            raise RuntimeError("Missing Right Code API key. Set IMAGE_API_KEY in .env.")

        image_data_url = self._encode_image_data_url(image)
        size = self.extra_params.get("size", f"{self.image_size[0]}x{self.image_size[1]}")
        payload = {
            "model": self.model_version,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": self.temperature,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if size:
            payload["size"] = size
        for key in ("n", "quality", "response_format"):
            if key in self.extra_params:
                payload[key] = self.extra_params[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = self.api_endpoint.rstrip("/")
        url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"

        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.is_error:
                raise RuntimeError(f"Right Code API {resp.status_code}: {resp.text[:500]}")
            data = resp.json()

        return self._parse_response(data)

    @staticmethod
    def _encode_image_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _parse_response(cls, data: dict) -> Image.Image:
        """Extract an image from common chat-completions response shapes."""
        direct = cls._parse_direct_image(data)
        if direct is not None:
            return direct

        choices = data.get("choices", [])
        if not choices:
            output = data.get("output", {})
            choices = output.get("choices", choices)
        if not choices:
            raise ValueError(f"No choices in Right Code response. Keys: {list(data.keys())}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        parsed = cls._parse_content(content)
        if parsed is not None:
            return parsed

        raise ValueError(f"Cannot parse image from Right Code response. Content preview: {str(content)[:300]}")

    @classmethod
    def _parse_direct_image(cls, data: dict) -> Optional[Image.Image]:
        for item in data.get("data", []):
            parsed = cls._parse_mapping_for_image(item)
            if parsed is not None:
                return parsed
        return cls._parse_mapping_for_image(data)

    @classmethod
    def _parse_content(cls, content) -> Optional[Image.Image]:
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    parsed = cls._parse_mapping_for_image(part)
                    if parsed is not None:
                        return parsed
                    if part.get("type") in {"text", "output_text"}:
                        parsed = cls._parse_text_for_image(part.get("text", ""))
                        if parsed is not None:
                            return parsed
                elif isinstance(part, str):
                    parsed = cls._parse_text_for_image(part)
                    if parsed is not None:
                        return parsed
        if isinstance(content, dict):
            return cls._parse_mapping_for_image(content)
        if isinstance(content, str):
            return cls._parse_text_for_image(content)
        return None

    @classmethod
    def _parse_mapping_for_image(cls, obj: dict) -> Optional[Image.Image]:
        for key in ("b64_json", "b64_image", "image_base64"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                return Image.open(io.BytesIO(base64.b64decode(val)))

        for key in ("url", "image", "image_url", "output_image"):
            val = obj.get(key)
            if isinstance(val, dict):
                val = val.get("url") or val.get("data") or val.get("b64_json")
            if isinstance(val, str) and val:
                parsed = cls._parse_text_for_image(val)
                if parsed is not None:
                    return parsed
        return None

    @classmethod
    def _parse_text_for_image(cls, text: str) -> Optional[Image.Image]:
        if not text:
            return None
        text = text.strip()
        if text.startswith("data:image/"):
            return _decode_data_url(text)
        if text.startswith("http://") or text.startswith("https://"):
            return _download_image(text)

        markdown_match = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", text)
        if markdown_match:
            return _download_image(markdown_match.group(1))

        url_match = re.search(r"https?://\S+", text)
        if url_match:
            return _download_image(url_match.group(0).rstrip(").,"))

        b64_match = re.search(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", text)
        if b64_match:
            return _decode_data_url(b64_match.group(0).replace("\n", ""))

        return cls._parse_class_grid(text)

    @staticmethod
    def _parse_class_grid(text: str) -> Optional[Image.Image]:
        """Convert a JSON class-index grid returned in text form to a mask."""
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        candidates = [cleaned]
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])

        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                decoded = decoded.get("grid") or decoded.get("mask") or decoded.get("classes")
            array = np.asarray(decoded)
            if array.ndim != 2 or array.size == 0 or not np.issubdtype(array.dtype, np.number):
                continue
            return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="L")
        return None
