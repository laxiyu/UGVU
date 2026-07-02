"""Qwen (千问) generator — Alibaba BaiLian (百炼) API wrapper.

API 端点: compatible-mode/v1 (OpenAI 兼容)

关键约束（经 API 测试确认）:
    - qwen-vl-max / qwen-vl-plus / qwen3.x 模型: 仅返回 TEXT，不支持直接输出图像
    - response_format 仅支持 "json_object" 和 "text"，不支持 "image"
    - qwen-image-2.0-pro: 专用图像生成模型，通过 images/generations 端点输出图像

因此本生成器采用"双轨"策略:
    1. generation 模式 (qwen-image-*) → 直接调用图像生成 API 输出 mask 图像
    2. chat 模式 (qwen-vl-*, qwen3.*) → 输出结构化文本 → 后处理为 mask 数组

环境变量:
    DASHSCOPE_API_KEY — 百炼控制台 API Key
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Literal, Optional

import numpy as np
from PIL import Image

from .base_generator import BaseGenerator

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ---------------------------------------------------------------------------
# Mask encoding / decoding helpers (for text-based mask output)
# ---------------------------------------------------------------------------

def _rle_decode(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode a Run-Length Encoding string into a 2D binary mask.

    RLE format: "length1 length2 length3 ..." (alternating 0-run, 1-run)
    """
    lengths = list(map(int, rle_str.strip().split()))
    mask = np.zeros(height * width, dtype=np.int64)
    pos = 0
    bit = 0
    for l in lengths:
        if bit == 1:
            mask[pos:pos + l] = 1
        pos += l
        bit ^= 1
    return mask.reshape(height, width)


def _rle_encode(mask: np.ndarray) -> str:
    """Encode a 2D binary mask as RLE string (row-major).

    格式: 交替的 0-run 和 1-run 长度，用空格分隔。
    始终以 0-run 开始（即使长度为 0）。
    """
    flat = mask.ravel()
    runs = []
    current = flat[0]
    count = 1
    for v in flat[1:]:
        if v == current:
            count += 1
        else:
            runs.append(count)
            current = v
            count = 1
    runs.append(count)

    # RLE 以 0-run 开始; 如果 mask 以 1 开头, 补一个 0-run
    if flat[0] == 1:
        runs = [0] + runs
    # 如果以 0 结尾, 补一个长度为 0 的 1-run 使成对 (解码安全)
    if len(runs) % 2 == 1:
        runs.append(0)

    return " ".join(str(r) for r in runs)


def _parse_json_mask(text: str, height: int, width: int) -> Optional[np.ndarray]:
    """Try to extract and parse a JSON mask from the model's text response.

    Supports multiple JSON formats:
        1. {"class_0_rle": "...", "class_1_rle": "..."}
        2. {"mask": [[row, col], ...]}  — list of foreground pixel coordinates
        3. {"shape": [H, W], "values": [[r1,c1], ...]}
    """
    # Find JSON block in markdown or plain text
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try finding { } directly
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = text[brace_start:brace_end + 1]
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Format 1: RLE per class (binary)
    if "class_0_rle" in data or "rle" in data:
        rle_key = "rle" if "rle" in data else "class_1_rle"
        rle_str = data.get(rle_key, "")
        if rle_str:
            return _rle_decode(rle_str, height, width)

    # Format 2: List of foreground pixels
    if "mask" in data and isinstance(data["mask"], list):
        mask = np.zeros((height, width), dtype=np.int64)
        valid_h = min(height, data.get("shape", [height, width])[0])
        valid_w = min(width, data.get("shape", [height, width])[1])
        for pt in data["mask"]:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                r, c = int(pt[0]), int(pt[1])
                if 0 <= r < valid_h and 0 <= c < valid_w:
                    mask[r, c] = 1
        return mask

    # Format 3: Matrix grid (compressed — e.g. 16x16 super-pixels)
    if "grid" in data:
        grid = np.array(data["grid"])
        scale_h = height // grid.shape[0]
        scale_w = width // grid.shape[1]
        mask = np.repeat(np.repeat(grid, scale_h, axis=0), scale_w, axis=1)
        mask = mask[:height, :width]
        return mask

    return None


def _parse_segmentation_mask(text: str, height: int, width: int) -> np.ndarray:
    """Main parsing function — tries multiple strategies to extract a mask.

    Returns:
        (H, W) int64 class-index array. Falls back to zeros if parsing fails.
    """
    h, w = height, width

    # Strategy 1: Try JSON parsing
    mask = _parse_json_mask(text, h, w)
    if mask is not None:
        return mask

    # Strategy 2: Try RLE in plain text
    # Look for "RLE:" or "rle:" followed by numbers
    rle_match = re.search(r'(?:RLE|rle)\s*[:=]\s*([\d\s]+)', text)
    if rle_match:
        try:
            return _rle_decode(rle_match.group(1), h, w)
        except Exception:
            pass

    # Strategy 3: Try detecting ASCII art mask (# = foreground, . = background)
    # Look for blocks of # and . characters
    ascii_lines = re.findall(r'^[#\.\s]+$', text, re.MULTILINE)
    if len(ascii_lines) > 5:  # at least 6 rows
        rows = []
        for line in ascii_lines:
            row = np.array([1 if c == '#' else 0 for c in line.strip()])
            if len(row) > 0:
                rows.append(row)
        if rows:
            ascii_mask = np.stack(rows[:min(len(rows), h)])
            if ascii_mask.shape[1] > 0:
                # Pad/crop to target size
                result = np.zeros((h, w), dtype=np.int64)
                rh = min(ascii_mask.shape[0], h)
                rw = min(ascii_mask.shape[1], w)
                result[:rh, :rw] = ascii_mask[:rh, :rw]
                return result

    # Strategy 4: Parse class-encoded grid from markdown table
    #   | 0 | 0 | 1 | 1 | 0 | ...
    lines = text.strip().split("\n")
    grid_rows = []
    for line in lines:
        cells = re.findall(r'\b(\d+)\b', line)
        if len(cells) > 4:  # at least 5 numbers = grid row
            grid_rows.append([int(c) for c in cells])

    if len(grid_rows) > 4:
        grid = np.array(grid_rows, dtype=np.int64)
        scale_h = h // grid.shape[0]
        scale_w = w // grid.shape[1]
        result = np.zeros((h, w), dtype=np.int64)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                rh = r * scale_h
                rw = c * scale_w
                result[rh:rh + scale_h, rw:rw + scale_w] = grid[r, c]
        return result

    # Fallback: return zeros
    return np.zeros((h, w), dtype=np.int64)


# ---------------------------------------------------------------------------
# Qwen generator
# ---------------------------------------------------------------------------

DEFAULT_MASK_PROMPT = """Analyze the input image and generate a pixel-wise segmentation mask.

Follow these EXACT rules:
1. Think about which pixels belong to each class.
2. OUTPUT YOUR RESULT AS A JSON OBJECT inside a ```json code block.
3. If this is a binary/foreground task, use format: {"rle": "length1 length2 length3 ..."}
   where the RLE alternates between background runs and foreground runs (row-major order).
4. If there are multiple classes, use: {"class_0_rle": "...", "class_1_rle": "...", ...}
5. If a pixel grid is easier, use: {"grid": [[0,1,0,...], [0,1,1,...], ...], "classes": ["bg", "fg"]}

IMPORTANT: Return ONLY the JSON block. No other text."""


class QwenGenerator(BaseGenerator):
    """Alibaba Qwen (千问) via BaiLian (百炼) 兼容模式 API.

    两种工作模式:
        chat (qwen-vl-max, qwen3.x) — 模型输出结构化文本 → 解析为 mask 图像
        generation (qwen-image-2.0-pro) — 直接调用 /images/generations 生成 mask 图像
    """

    def __init__(
        self,
        api_endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "",
        model_version: str = "qwen-vl-max",
        api_mode: Literal["chat", "generation"] = "chat",
        **kwargs,
    ):
        # 从环境变量读取 API Key
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("ALIBABA_CLOUD_API_KEY", "")
        model_name = kwargs.pop("model_name", model_version)
        super().__init__(
            model_name=model_name,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )

        # 自动检测模式
        if model_version.startswith("qwen-image"):
            self.api_mode = "generation"
        elif model_version.startswith(("qwen-vl", "qwen3.")):
            self.api_mode = "chat"
        else:
            self.api_mode = api_mode

        # 可选 colormap, 用于 chat 模式输出 RGB mask 图像
        self.colormap: Optional[np.ndarray] = None

    # ======================================================================
    # 统一调用入口
    # ======================================================================

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if self.api_mode == "generation":
            return self._call_generation_api(image, prompt, seed)
        else:  # chat
            return self._call_chat_api(image, prompt, seed)

    # ======================================================================
    # Mode 1: Image Generation (qwen-image-2.0-pro via DashScope API)
    # ======================================================================

    def _call_generation_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """调用 qwen-image-2.0-pro 图像生成 API.

        DashScope 原生端点:
            POST /api/v1/services/aigc/image-generation/generation

        支持:
            - 文生图 (text-to-image): 仅传 prompt
            - 图生图 (image-to-image): 传 prompt + 参考图 (通过 extra_params)

        API 文档: https://help.aliyun.com/zh/model-studio/developer-reference/image-generation
        """
        if not HAS_HTTPX:
            raise ImportError("httpx is required. Install with: pip install httpx")

        import base64

        # 构建 request body (DashScope 原生格式)
        input_data = {"prompt": prompt}

        # DashScope qwen-image generation is primarily text-to-image. Some
        # deployments may support reference images, so keep it opt-in.
        if (
            image is not None
            and "compatible" not in self.api_endpoint
            and self.extra_params.get("use_reference_image", False)
        ):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            input_data["reference_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload: dict = {
            "model": self.model_version,
            "input": input_data,
            "parameters": {
                "size": f"{self.image_size[0]}*{self.image_size[1]}",
                "n": 1,
            },
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 确定端点 URL
        if "/image-generation/" in self.api_endpoint:
            # 已经是完整端点路径，直接使用
            url = self.api_endpoint
        elif "compatible" in self.api_endpoint:
            # 兼容模式走 chat/completions (仅文本 prompt)
            url = f"{self.api_endpoint.rstrip('/v1')}/chat/completions"
            # 切回纯文本 chat 格式
            payload = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": 4096,
            }
        else:
            # 基 URL → 拼接图像生成端点
            base = self.api_endpoint.rstrip('/')
            url = f"{base}/api/v1/services/aigc/image-generation/generation"

        with httpx.Client(timeout=self.timeout_sec, trust_env=False) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.is_error:
                raise RuntimeError(f"Qwen API {resp.status_code}: {resp.text[:500]}")
            data = resp.json()

        return self._parse_generation_response(data)

    def _parse_generation_response(self, data: dict) -> Image.Image:
        """解析图像生成响应 (DashScope 原生格式)."""
        # DashScope 格式: output.results[].{url, b64_image}
        output = data.get("output", {})
        for result in output.get("results", []):
            # 优先 URL
            url = result.get("url", "")
            if url:
                from .doubao import _download_image
                try:
                    return _download_image(url)
                except Exception:
                    pass
            # 其次 base64
            for key in ("b64_image", "image", "b64_json"):
                b64 = result.get(key, "")
                if b64:
                    import base64
                    return Image.open(io.BytesIO(base64.b64decode(b64)))

        # OpenAI 兼容格式: data[].{b64_json, url} (兼容模式回退)
        for item in data.get("data", []):
            for key in ("b64_json", "url"):
                val = item.get(key, "")
                if val:
                    if key == "url":
                        from .doubao import _download_image
                        return _download_image(val)
                    else:
                        import base64
                        return Image.open(io.BytesIO(base64.b64decode(val)))

        raise ValueError(
            f"Cannot parse image from generation response.\n"
            f"Keys: {list(data.keys())}\n"
            f"Has output: {'output' in data}\n"
            f"Has data: {'data' in data}"
        )

    # ======================================================================
    # Mode 2: Chat Completions (qwen-vl-max / qwen3.x — 文本输出)
    # ======================================================================

    def _call_chat_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """调用 chat/completions 端点 → 获取结构化文本 → 解析为 mask 图像.

        qwen-vl-max 等 VL 模型只能返回 TEXT。
        本方法:
            1. 将用户 prompt 与一个结构化输出指令拼接
            2. 发送给模型获取文本响应
            3. 解析文本响应中的结构 (JSON / RLE / 网格) 为 mask 数组
            4. 返回 mask 作为 PIL Image
        """
        if not HAS_HTTPX:
            raise ImportError("httpx is required. Install with: pip install httpx")

        import base64

        # 编码输入图像
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_data_url = f"data:image/png;base64,{img_b64}"

        h, w = image.size[1], image.size[0]

        # 构造 prompt: 用户原始 prompt + 结构化输出指令
        structured_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Output a JSON object containing the segmentation mask data.\n"
            f"The image size is {w}x{h} pixels.\n"
            f"Use this format: {{\"grid\": [[row0_values], [row1_values], ...], \"classes\": [\"class0\", \"class1\", ...]}}\n"
            f"Where each row value is the class index (0-based).\n"
            f"Do not return an all-background, all-unknown, or all-255 grid.\n"
            f"To save bandwidth, use a reduced grid of {max(8, h//32)}x{max(8, w//32)} cells.\n"
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

        payload: dict = {
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

        return self._parse_chat_response(data, h, w, colormap=self.colormap)

    def _parse_chat_response(
        self, data: dict, height: int, width: int, colormap: Optional[np.ndarray] = None
    ) -> Image.Image:
        """解析 chat 响应文本 → 重建为 RGB mask 图像.

        如果提供了 colormap, 使用色图着色 class mask 为 RGB 图像,
        以便下游 ColormapDecoder 能正确匹配.
        """
        choices = data.get("choices", [])
        if not choices:
            output = data.get("output", {})
            choices = output.get("choices", choices)

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

        # 解析 mask (class indices)
        mask = _parse_segmentation_mask(content, height, width)

        # 使用 colormap 着色为 RGB
        if colormap is not None and mask.max() < len(colormap):
            h, w = mask.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(len(colormap)):
                rgb[mask == c] = colormap[c]
            return Image.fromarray(rgb)
        else:
            img = np.clip(mask, 0, 255).astype(np.uint8)
            return Image.fromarray(img, mode="L")

    # ======================================================================
    # 工具方法
    # ======================================================================

    @staticmethod
    def _encode_image_b64(image: Image.Image) -> str:
        """编码 PIL Image 为 base64 字符串."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        import base64
        return base64.b64encode(buf.getvalue()).decode("utf-8")



