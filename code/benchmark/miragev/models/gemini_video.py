from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import ModelAdapter, ModelOutput
from .registry import register
from ..utils import ensure_dir


@register("gemini_video")
class GeminiVideoAdapter(ModelAdapter):
    """Gemini adapter that sends the *video file* directly (native video understanding).

    Uses the Google Gemini API 'Files' upload then includes the uploaded file reference
    in generate_content.

    Requires:
      export GEMINI_API_KEY=...
      pip install google-genai

    Notes:
      - Uploads are cached by absolute video_path in a JSON cache under cache_dir.
      - This adapter ignores sampled frames (images) and uses video_path instead.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        cache_dir: str = ".cache/gemini",
    ):
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError("Install Gemini SDK: pip install google-genai") from e

        self.genai = genai
        self.client = genai.Client()
        self.model = model
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self.cache_path = os.path.join(cache_dir, "uploaded_files.json")
        self._cache: Dict[str, Any] = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _wait_for_file_active(self, myfile, max_wait: int = 60):
        """Wait for file to be in ACTIVE state before use."""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Refresh file status
            try:
                myfile = self.client.files.get(name=myfile.name)
                state = getattr(myfile, "state", None)
                if state == "ACTIVE":
                    return myfile
                elif state in ["FAILED", "EXPIRED"]:
                    raise RuntimeError(f"File upload failed with state: {state}")
            except Exception as e:
                # If get fails, try to check state directly
                state = getattr(myfile, "state", None)
                if state == "ACTIVE":
                    return myfile
                if state in ["FAILED", "EXPIRED"]:
                    raise RuntimeError(f"File upload failed with state: {state}")
            
            time.sleep(1)
        
        raise RuntimeError(f"File did not become ACTIVE within {max_wait} seconds")

    def _upload_if_needed(self, video_path: str):
        key = os.path.abspath(video_path)
        if key in self._cache and self._cache[key].get("name"):
            # Reconstruct file reference
            name = self._cache[key]["name"]
            # Files API returns a file object when calling get
            try:
                myfile = self.client.files.get(name=name)
                # Check if file is still active
                state = getattr(myfile, "state", None)
                if state == "ACTIVE":
                    return myfile
                # If not active, re-upload
            except Exception:
                # fallback: re-upload
                pass

        myfile = self.client.files.upload(file=video_path)
        # Wait for file to be active before caching
        myfile = self._wait_for_file_active(myfile)
        # Cache minimal metadata to re-fetch later
        self._cache[key] = {"name": getattr(myfile, "name", None), "uri": getattr(myfile, "uri", None)}
        self._save_cache()
        return myfile

    def predict(
        self,
        *,
        instructions: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        if not video_path:
            raise ValueError("GeminiVideoAdapter requires video_path (local file).")

        myfile = self._upload_if_needed(video_path)

        full_prompt = instructions + "\n\n" + prompt
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[myfile, full_prompt],
        )
        text = getattr(resp, "text", "") or ""
        return ModelOutput(raw_text=text.strip(), meta={"adapter": "gemini_video", "model": self.model})
