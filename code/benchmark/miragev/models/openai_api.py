from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import ModelAdapter, ModelOutput
from .registry import register


def _img_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


@register("openai")
class OpenAIFramesAdapter(ModelAdapter):
    """OpenAI adapter that evaluates video by providing sampled frames as image inputs.

    Uses the OpenAI Responses API with input items containing:
      - input_text
      - repeated input_image entries (one per frame)

    Requires:
      export OPENAI_API_KEY=...
      pip install openai
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        max_output_tokens: int = 128,
        image_format: str = "JPEG",
        image_quality: int = 85,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Install OpenAI python SDK: pip install openai") from e

        print(f"Initializing OpenAI adapter for model: {model}")
        print(f"Max output tokens: {max_output_tokens}")
        self.client = OpenAI()
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.image_format = image_format
        self.image_quality = image_quality
        self._last_request_time = 0.0

    def predict(
        self,
        *,
        instructions: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        content = [{"type": "input_text", "text": instructions + "\n\n" + prompt}]
        for img in (images or []):
            content.append({"type": "input_image", "image_url": _img_to_data_url(img, self.image_format, self.image_quality)})

        # Slow down requests to avoid connection issues (1 second delay)
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 0.5:
            time.sleep(0.5 - time_since_last)
        self._last_request_time = time.time()

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self.max_output_tokens,
        )
        
        text = getattr(resp, "output_text", "") or ""
        if not text and hasattr(resp, "output"):
            # Try alternative response structure
            output = resp.output
            if isinstance(output, list) and len(output) > 0:
                first_item = output[0]
                if hasattr(first_item, "text"):
                    text = first_item.text
                elif isinstance(first_item, dict) and "text" in first_item:
                    text = first_item["text"]
        
        return ModelOutput(raw_text=text.strip(), meta={"adapter": "openai", "model": self.model})
