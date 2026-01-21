from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import ModelAdapter, ModelOutput
from .registry import register


@register("hf")
class HuggingFaceMultiImageAdapter(ModelAdapter):
    """Reference adapter for Hugging Face multimodal chat models.

    IMPORTANT:
      Multimodal chat formatting varies across models. This adapter tries a robust,
      widely-compatible path:

      - Insert an image placeholder token (default: <image>) once per frame
      - Use processor.apply_chat_template if available
      - Call model.generate and decode

    For some models, you may need:
      --hf_image_token "<image>"  (or the model's special image token)
      --hf_chat_template "auto"   (default) or "none"
      --hf_device_map "auto"
      --hf_dtype "auto" | "bfloat16" | "float16"

    If a specific model requires custom formatting, add a new adapter under
    miragev/models/ and register it in registry.py.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        trust_remote_code: bool = True,
        image_token: str = "<image>",
        chat_template: str = "auto",  # auto | none
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.image_token = image_token
        self.chat_template = chat_template

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

            # Try Vision2Seq first, fallback to CausalLM
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name_or_path,
                    device_map=device_map,
                    torch_dtype=dtype if dtype != "auto" else None,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map=device_map,
                    torch_dtype=dtype if dtype != "auto" else None,
                    trust_remote_code=trust_remote_code,
                )

            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Hugging Face adapter. "
                "Install extras: pip install -r requirements.txt (or pip install .[hf]) "
                "and ensure your model supports vision inputs."
            ) from e

    def _build_text(self, instructions: str, prompt: str, images: Optional[List[Image.Image]]) -> str:
        # Put images first via placeholder tokens; many models expect this.
        if images and self.image_token:
            prefix = (self.image_token + "\n") * len(images)
            full = prefix + prompt
        else:
            full = prompt
        # Some models want system+user messages; others accept plain text.
        if self.chat_template == "none":
            return instructions + "\n\n" + full
        # auto: use chat template if processor supports it
        if hasattr(self.processor, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": full},
            ]
            try:
                return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fall back to plain concatenation.
                return instructions + "\n\n" + full
        return instructions + "\n\n" + full

    def predict(
        self,
        *,
        instructions: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        text = self._build_text(instructions, prompt, images)

        # Processor call signatures vary slightly; try a couple forms.
        kwargs = {"return_tensors": "pt"}
        try:
            inputs = self.processor(text=text, images=images, **kwargs)
        except TypeError:
            inputs = self.processor(text=[text], images=images, **kwargs)

        # Move tensors to model device
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": max(self.temperature, 1e-5),
            "top_p": self.top_p,
        }

        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Some models return [prompt + answer]; decode and try to strip prompt.
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)
        raw = decoded[0] if decoded else ""

        # Heuristic: take last line after 'Answer:' if present
        if "Answer:" in raw:
            raw = raw.split("Answer:")[-1].strip()

        return ModelOutput(raw_text=raw.strip(), meta={"adapter": "hf", "model": self.model_name_or_path})
