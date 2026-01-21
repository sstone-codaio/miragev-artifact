from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class ModelOutput:
    raw_text: str
    meta: Dict[str, Any]


class ModelAdapter(ABC):
    """Abstract model adapter.

    Implementations must return the model's raw text response.
    Parsing into a discrete answer is handled by the evaluator for consistency.
    """

    @abstractmethod
    def predict(
        self,
        *,
        instructions: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        raise NotImplementedError
