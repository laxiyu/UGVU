"""Doubao (豆包) generator — ByteDance Volcano Ark API wrapper.

Uses the Vision API to generate segmentation masks / depth maps from
an input image + prompt.

API docs: https://www.volcengine.com/docs/82379
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
from PIL import Image

from .base_generator import BaseGenerator

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# Seedream 最小分辨率: 1920×1920 = 3,686,400 像素
_SEEDREAM_MIN_PIXELS = 3_686_400


class DoubaoGenerator(BaseGenerator):
    """ByteDance Doubao / Seedream generator — Volcano Ark API wrapper.

    Supports three modes via `api_mode`:
      - "vision" (default): calls /chat/completions, expects image output
      - "vision_chat": calls /chat/completions, expects text → parsed to mask
      - "seedream": calls /v3/images/generations (text-to-image, prompt-only)

    Environment variables:
        DOUBAO_API_KEY — API key for authentication.
    """

    def __init__(
        self,
        api_endpoint: str = "https://ark.cn-beijing.volces.com/api/v3",
        api_key: str = "",
        model_version: str = "doubao-vision-pro-32k",
        api_mode: str = "vision",  # "vision" | "vision_chat" | "seedream"
        **kwargs,
    ):
        api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.api_mode = api_mode
        self.colormap = None  # 可从 pipeline 设置，用于 vision_chat 模式的 mask 着色
        # build_sampler 会传入 model_name=name, 用它而不是硬编码 "doubao"
        model_name = kwargs.pop("model_name", "doubao")
        super().__init__(
            model_name=model_name,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )
        # 自动修正 Seedream 的最小分辨率
        if self.api_mode == "seedream":
            w, h = self.image_size
            pixels = w * h
            if pixels < _SEEDREAM_MIN_PIXELS:
                # 等比例放大到满足最小像素要求
                scale = (_SEEDREAM_MIN_PIXELS / pixels) ** 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                # 取整到最近的 128 倍数（常用对齐）
                new_w = ((new_w + 127) // 128) * 128
                new_h = ((new_h + 127) // 128) * 128
                # 确保不小于最小像素数
                while new_w * new_h < _SEEDREAM_MIN_PIXELS:
                    new_w += 128
                    new_h += 128
                self.image_size = (new_w, new_h)
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[{self.model_name}] Seedream minimum resolution: "
                    f"adjusted image_size from {w}x{h} to {new_w}x{new_h}"
                )

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Route to the correct API based on api_mode."""
        if not HAS_HTTPX:
            raise ImportError("httpx is required for DoubaoGenerator. Install with: pip install httpx")

        if self.api_mode == "seedream":
            return self._call_seedream_api(prompt, seed)
        elif self.api_mode == "vision_chat":
            return self._call_vision_chat_api(image, prompt, seed)
        else:
            return self._call_vision_api(image, prompt, seed)

    # ======================================================================
    # Vision mode (Doubao Vision — /chat/completions)
    # ======================================================================

    def _call_vision_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Call Doubao Vision API (image+text → image).

        The API expects:
            - model: the model version string
            - messages: a chat-format list with image + text
            - parameters: optional seed, temperature, etc.
        """
        # Prepare image as base64 data URL
        img_bytes = self._encode_image(image)

        # Build request payload
        payload = {
            "model": self.model_version,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_bytes},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "parameters": {
                "temperature": self.temperature,
            },
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Make request
        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(
                f"{self.api_endpoint}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data)

    # ======================================================================
    # Seedream mode (text-to-image — /v3/images/generations)
    # ======================================================================

    def _call_seedream_api(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Call Seedream text-to-image API via /v3/images/generations.

        Seedream is a pure text-to-image model — it does NOT accept an
        input image. The `image` parameter in `generate()` is ignored.

        API docs: https://www.volcengine.com/docs/6791
        """
        w, h = self.image_size
        payload = {
            "model": self.model_version,  # 这里传入 Endpoint ID
            "prompt": prompt,
            "n": 1,
            "size": f"{w}x{h}",
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(
                f"{self.api_endpoint}/images/generations",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_seedream_response(data)

    @staticmethod
    def _parse_seedream_response(data: dict) -> Image.Image:
        """Parse Seedream response — extract image URL and download."""
        results = data.get("data", [])
        if not results:
            raise ValueError(f"No image data in Seedream response: {data}")

        img_info = results[0]
        url = img_info.get("url", "")
        b64 = img_info.get("b64_json", "")

        if url:
            return _download_image(url)
        elif b64:
            import base64
            return Image.open(io.BytesIO(base64.b64decode(b64)))

        raise ValueError(f"Cannot extract image from Seedream response: {list(img_info.keys())}")

    # ======================================================================
    # Vision Chat mode — for text-returning VL models (doubao-1.5-vision-pro 等)
    # ======================================================================

    def _call_vision_chat_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Call Doubao Vision Chat API (image+text → text → parsed mask).

        Doubao-1.5-Vision-Pro 等 VL 模型只能返回 TEXT。
        本方法：
            1. 发送图片 + 结构化 prompt（要求模型输出 JSON 格式的 mask 数据）
            2. 解析回复中的 JSON（grid/RLE 格式）
            3. 渲染为 RGB mask 图像返回
        """
        import base64

        # 编码输入图像
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_data_url = f"data:image/png;base64,{img_b64}"

        w, h = self.image_size if hasattr(self, 'image_size') else image.size
        # 如果 image_size 是 tuple/list 但 image 尺寸不同，用 image 的实际尺寸
        if image.size != (w, h) if isinstance(w, int) else True:
            w, h = image.size

        # 构造结构化输出 prompt
        grid_h = max(8, h // 32)
        grid_w = max(8, w // 32)
        structured_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Output a JSON object containing the segmentation mask data.\n"
            f"The image size is {w}x{h} pixels.\n"
            f"Use this format: {{\"grid\": [[row0_values], [row1_values], ...], \"classes\": [\"class0\", \"class1\", ...]}}\n"
            f"Where each row value is the class index (0-based).\n"
            f"Do not return an all-background, all-unknown, or all-255 grid.\n"
            f"To save bandwidth, use a reduced grid of {grid_h}x{grid_w} cells.\n"
            f"Wrap the JSON in ```json ... ``` code block."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                    {"type": "text", "text": structured_prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_version,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 8192,
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(
                f"{self.api_endpoint}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_vision_chat_response(data, h, w)

    def _parse_vision_chat_response(self, data: dict, height: int, width: int) -> Image.Image:
        """Parse vision chat text response → reconstruct as RGB mask image.

        如果提供了 colormap，用色图为 mask 着色以便下游 ColormapDecoder 匹配。
        """
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response. Keys: {list(data.keys())}")

        content = choices[0].get("message", {}).get("content", "")

        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
            content = "\n".join(texts)

        if not isinstance(content, str):
            content = str(content)

        # 解析 mask（class indices）
        mask = _parse_segmentation_mask(content, height, width)

        # 使用 colormap 着色为 RGB
        if self.colormap is not None and mask.max() < len(self.colormap):
            h, w = mask.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(len(self.colormap)):
                rgb[mask == c] = self.colormap[c]
            return Image.fromarray(rgb)
        else:
            img_arr = np.clip(mask, 0, 255).astype(np.uint8)
            return Image.fromarray(img_arr, mode="L")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL Image as a base64 data URL string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _parse_response(data: dict) -> Image.Image:
        """Extract generated image from API response.

        The response contains either:
            - A generated image URL (needs download)
            - Base64-encoded image data in the content
        """
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Try to parse as image URL or base64
        if isinstance(content, list):
            # Multi-part content — search for image
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        return _decode_data_url(url)
                    elif url.startswith("http"):
                        return _download_image(url)
                elif part.get("type") == "image":
                    return _decode_data_url(part.get("image", ""))
            raise ValueError(f"No image part found in response content: {content}")

        elif isinstance(content, str):
            # Could be a URL string or base64 inline
            if content.startswith("data:"):
                return _decode_data_url(content)
            elif content.startswith("http"):
                return _download_image(content)
            # Sometimes models return markdown image links
            if "![" in content:
                import re
                match = re.search(r'\((https?://[^\)]+)\)', content)
                if match:
                    return _download_image(match.group(1))

        raise ValueError(f"Cannot parse image from response. Content preview: {str(content)[:200]}")


# ------------------------------------------------------------------
# Shared image decoding helpers
# ------------------------------------------------------------------

import base64 as _base64


def _decode_data_url(url: str) -> Image.Image:
    """Decode a `data:image/...;base64,...` URL into a PIL Image."""
    header, encoded = url.split(",", 1)
    return Image.open(io.BytesIO(_base64.b64decode(encoded)))


def _download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    import httpx
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))


# ------------------------------------------------------------------
# Mask parsing helpers — for vision_chat mode (text → structured mask)
# 复用 qwen.py 中的成熟解析逻辑
# ------------------------------------------------------------------

from .qwen import _parse_segmentation_mask  # noqa: E402, F401


