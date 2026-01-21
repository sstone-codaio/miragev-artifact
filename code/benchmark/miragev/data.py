from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

from datasets import load_dataset

from .schema import NormalizedExample


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _coerce_video_path(row: Dict[str, Any]) -> Optional[str]:
    # Preferred nested schema
    vp = _get(row, "video.path") or row.get("video_path") or row.get("path")
    if isinstance(vp, str) and vp.strip():
        return vp

    # HF 'Video' feature sometimes yields dict-like values, e.g. {"path": "...", "bytes": None}
    v = row.get("video")
    if isinstance(v, dict):
        p = v.get("path")
        if isinstance(p, str) and p.strip():
            return p
    return None


def normalize_row(row: Dict[str, Any]) -> NormalizedExample:
    example_id = row.get("example_id") or row.get("id") or row.get("_id") or ""

    pair_id = _get(row, "pairing.pair_id") or row.get("pair_id")
    pair_role = _get(row, "pairing.pair_role") or row.get("pair_role")

    q = row.get("question", {}) if isinstance(row.get("question"), dict) else {}
    question_id = q.get("question_id") or row.get("question_id")

    question_type = (q.get("type") or row.get("question_type") or "mcq").lower()
    prompt = q.get("prompt") or row.get("prompt") or ""

    options = q.get("options") or row.get("options")
    if options is not None and not isinstance(options, list):
        options = None

    answer_index = q.get("answer_index") if "answer_index" in q else row.get("answer_index")
    if answer_index is None:
        # Some datasets store the correct option letter or text
        ans = q.get("answer") or row.get("answer")
        if isinstance(ans, int):
            answer_index = ans
        else:
            answer_index = None

    answer_text = None
    ans2 = q.get("answer") or row.get("answer")
    if isinstance(ans2, str):
        answer_text = ans2

    domain = row.get("domain") or _get(row, "physics.module") or _get(row, "physics.domain")
    module = _get(row, "physics.module") or row.get("module") or row.get("domain")

    cw = _get(row, "temporal.critical_window") or row.get("critical_window")
    critical_window: Optional[Tuple[float, float]] = None
    if isinstance(cw, (list, tuple)) and len(cw) == 2:
        try:
            critical_window = (float(cw[0]), float(cw[1]))
        except Exception:
            critical_window = None

    video_path = _coerce_video_path(row)

    return NormalizedExample(
        example_id=str(example_id),
        video_path=video_path,
        pair_id=str(pair_id) if pair_id is not None else None,
        pair_role=str(pair_role) if pair_role is not None else None,
        question_id=str(question_id) if question_id is not None else None,
        question_type=question_type,
        prompt=str(prompt),
        options=options,
        answer_index=int(answer_index) if isinstance(answer_index, int) else None,
        answer_text=answer_text,
        domain=str(domain) if domain is not None else None,
        module=str(module) if module is not None else None,
        critical_window=critical_window,
        raw=row,
    )


def load_miragev(dataset: str, split: str = "test", data_dir: Optional[str] = None):
    """Load MIRAGE-V.

    Args:
      dataset:
        - HF dataset name like "ORG/MIRAGE-V"
        - OR local path to JSON/JSONL: "./miragev_test.jsonl"
      split: HF split, or ignored for local files.
      data_dir: passed to load_dataset for some HF datasets.

    Returns: Hugging Face Dataset object
    """
    if os.path.exists(dataset):
        # Local file
        return load_dataset("json", data_files=dataset, split="train")
    return load_dataset(dataset, split=split, data_dir=data_dir)
