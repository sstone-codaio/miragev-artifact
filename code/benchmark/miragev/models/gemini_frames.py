from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import ModelAdapter, ModelOutput
from .registry import register


@register("gemini_frames")
class GeminiFramesAdapter(ModelAdapter):
    """
    Gemini adapter that evaluates a video by providing sampled frames as inline images,
    mirroring the OpenAI frame-sampling approach.

    This uses google-genai and types.Part.from_bytes(...) for each image frame.
    """
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        image_format: str = "JPEG",
        image_quality: int = 85,
    ):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:
            raise RuntimeError("Install Gemini SDK: pip install google-genai") from e

        self.genai = genai
        self.types = types
        self.client = genai.Client()
        self.model = model
        self.image_format = image_format
        self.image_quality = image_quality

    def _pil_to_part(self, img: Image.Image):
        buf = io.BytesIO()
        fmt = self.image_format.upper()
        if fmt in ("JPG", "JPEG"):
            img.save(buf, format="JPEG", quality=self.image_quality)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"
        return self.types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)

    def predict(
        self,
        *,
        instructions: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        # For text_only modes, images may be None; allow it.
        parts: List[Any] = []

        # Put frames first in temporal order (like OpenAI does)
        if images:
            for img in images:
                parts.append(self._pil_to_part(img))

        # Append the text last
        full_prompt = instructions + "\n\n" + prompt
        parts.append(full_prompt)

        # generate_content accepts mixed parts (text + inline media)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=parts,
        )
        text = getattr(resp, "text", "") or ""
        return ModelOutput(
            raw_text=text.strip(),
            meta={"adapter": "gemini_frames", "model": self.model, "n_images": len(images or [])},
        )
