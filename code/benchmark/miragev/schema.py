from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NormalizedExample:
    example_id: str
    video_path: Optional[str]
    pair_id: Optional[str]
    pair_role: Optional[str]
    question_id: Optional[str]
    question_type: str  # mcq | yesno
    prompt: str
    options: Optional[List[str]]
    answer_index: Optional[int]
    answer_text: Optional[str]
    domain: Optional[str]
    module: Optional[str]
    critical_window: Optional[Tuple[float, float]]
    raw: Dict[str, Any]
