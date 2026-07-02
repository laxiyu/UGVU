"""Flux generator — Black Forest Labs API wrapper (optional).

Uses the Flux Pro API for image generation / editing tasks.

API docs: https://docs.bfl.ml/
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

from PIL import Image

from .base_generator import BaseGenerator

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class FluxGenerator(BaseGenerator):
    """Black Forest Labs Flux Pro API wrapper.

    Sends an image + text prompt to Flux for image-to-image generation
    (e.g. segmentation mask overlay, depth stylization).

    Environment variables:
        FLUX_API_KEY — API key for BFL authentication.
    """

    def __init__(
        self,
        api_endpoint: str = "https://api.bfl.ml",
        api_key: str = "",
        model_version: str = "flux-pro-1.1",
        **kwargs,
    ):
        api_key = api_key or os.environ.get("FLUX_API_KEY", "")
        super().__init__(
            model_name="flux",
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )

    def _call_api(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Call Flux image-to-image API.

        BFL uses a polling workflow:
            1. POST /v1/flux-pro-1.1/image-to-image → get a task ID
            2. Poll GET /v1/get_result?id=... until status is "Ready"
            3. Download result image
        """
        if not HAS_HTTPX:
            raise ImportError("httpx is required for FluxGenerator. Install with: pip install httpx")

        import base64

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        headers = {
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Submit
        payload = {
            "prompt": prompt,
            "image": img_b64,
            "image_format": "png",
            "output_format": "png",
        }
        if seed is not None:
            payload["seed"] = seed

        with httpx.Client(timeout=self.timeout_sec) as client:
            submit_resp = client.post(
                f"{self.api_endpoint}/v1/{self.model_version}/image-to-image",
                json=payload,
                headers=headers,
            )
            submit_resp.raise_for_status()
            task_id = submit_resp.json()["id"]

            # Poll
            for _ in range(60):  # up to ~60s on a 1s poll interval
                time.sleep(1)
                poll_resp = client.get(
                    f"{self.api_endpoint}/v1/get_result",
                    params={"id": task_id},
                    headers=headers,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("status", "")
                if status == "Ready":
                    result_url = poll_data["result"]["sample"]
                    result_resp = client.get(result_url)
                    result_resp.raise_for_status()
                    return Image.open(io.BytesIO(result_resp.content))
                elif status in ("Failed", "Error"):
                    raise RuntimeError(f"Flux task {task_id} failed: {poll_data}")

        raise TimeoutError(f"Flux task {task_id} timed out after 60 polling attempts")
