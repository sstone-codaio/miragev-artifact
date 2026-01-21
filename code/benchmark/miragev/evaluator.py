from __future__ import annotations

import os

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .data import normalize_row
from .metrics import compute_metrics
from .prompts import build_prompt
from .schema import NormalizedExample
from .utils import now_iso
from .video import FrameSamplingConfig, extract_frames
from .models.base import ModelAdapter


_MC_LETTER_RE = re.compile(r"\b([A-Z])\b")
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_prediction(question_type: str, raw_text: str, options: Optional[List[str]]) -> Optional[int]:
    qt = (question_type or "mcq").lower()
    txt = (raw_text or "").strip()

    if qt == "yesno":
        # YES => 1, NO => 0 (arbitrary convention)
        if _YES_RE.search(txt) and not _NO_RE.search(txt):
            return 1
        if _NO_RE.search(txt) and not _YES_RE.search(txt):
            return 0
        # If both appear or neither: None
        return None

    # MCQ: accept letter or option text
    if options:
        # First try letter parsing
        m = _MC_LETTER_RE.search(txt[:20])  # usually very short
        if m:
            letter = m.group(1).upper()
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(options):
                return idx

        # Try match option text (lowercase)
        low = txt.lower()
        for i, opt in enumerate(options):
            if opt and opt.lower() in low:
                return i

    return None


def is_correct(ex: NormalizedExample, pred_index: Optional[int]) -> Optional[bool]:
    if pred_index is None:
        return False
    qt = ex.question_type.lower()
    if qt == "yesno":
        # If dataset uses answer_index, use it. Otherwise map answer_text.
        if ex.answer_index is not None:
            return pred_index == ex.answer_index
        if ex.answer_text:
            a = ex.answer_text.strip().lower()
            if a in ("yes", "true", "1"):
                return pred_index == 1
            if a in ("no", "false", "0"):
                return pred_index == 0
        return None
    # MCQ
    if ex.answer_index is not None:
        return pred_index == ex.answer_index
    if ex.answer_text and ex.options:
        try:
            idx = ex.options.index(ex.answer_text)
            return pred_index == idx
        except Exception:
            return None
    return None


def run_evaluation(
    dataset_rows: List[Dict[str, Any]],
    adapter: ModelAdapter,
    *,
    mode: str = "full_video",  # full_video | text_only | single_frame | shuffled_frames
    num_frames: int = 8,
    target_fps: Optional[float] = None,
    seed: int = 0,
    focus_critical_window: bool = False,
    critical_window_boost: int = 0,
    max_examples: Optional[int] = None,
    video_root: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = mode.lower()
    preds: List[Dict[str, Any]] = []

    rows_iter = dataset_rows[: max_examples] if max_examples else dataset_rows

    for row in tqdm(rows_iter, desc="Evaluating"):
        ex = normalize_row(row)

        resolved_video_path = ex.video_path
        if video_root and resolved_video_path and not os.path.isabs(resolved_video_path):
            resolved_video_path = os.path.join(video_root, resolved_video_path)


        instructions, prompt = build_prompt(ex.question_type, ex.prompt, ex.options)

        frames = None
        if mode == "text_only":
            frames = None
        else:
            if not resolved_video_path:
                raise ValueError(f"Missing video_path for example {ex.example_id}")
            cfg = FrameSamplingConfig(
                num_frames=num_frames,
                target_fps=target_fps,
                seed=seed,
                shuffle=(mode == "shuffled_frames"),
                single_frame=(mode == "single_frame"),
                focus_critical_window=focus_critical_window,
                critical_window_boost=critical_window_boost,
            )
            frames = extract_frames(resolved_video_path, cfg, ex.critical_window)

        out = adapter.predict(
            instructions=instructions,
            prompt=prompt,
            images=frames,
            video_path=resolved_video_path,
            extra={"example_id": ex.example_id, "pair_id": ex.pair_id, "pair_role": ex.pair_role},
        )

        pred_idx = parse_prediction(ex.question_type, out.raw_text, ex.options)
        correct = is_correct(ex, pred_idx)

        # Debug: print first few examples
        if len(preds) < 3:
            print(f"\n[DEBUG Example {len(preds)}]")
            print(f"  Example ID: {ex.example_id}")
            print(f"  Question: {ex.prompt}")
            print(f"  Question Type: {ex.question_type}")
            print(f"  Answer Index: {ex.answer_index}")
            print(f"  Answer Text: {ex.answer_text}")
            print(f"  Raw Output: {repr(out.raw_text)}")
            print(f"  Parsed Index: {pred_idx}")
            print(f"  Is Correct: {correct}")

        preds.append(
            {
                "ts": now_iso(),
                "example_id": ex.example_id,
                "pair_id": ex.pair_id,
                "pair_role": ex.pair_role,
                "question_id": ex.question_id,
                "question_type": ex.question_type,
                "module": ex.module,
                "domain": ex.domain,
                "video_path": resolved_video_path,
                "mode": mode,
                "num_frames": None if mode == "text_only" else (len(frames) if frames is not None else 0),
                "raw_output": out.raw_text,
                "pred_index": pred_idx,
                "is_correct": bool(correct) if correct is not None else False,
                "meta": out.meta,
            }
        )

    metrics = compute_metrics(preds)
    return preds, {
        "n": metrics.n,
        "accuracy": metrics.accuracy,
        "pair_accuracy": metrics.pair_accuracy,
        "by_module": metrics.by_module,
        "by_domain": metrics.by_domain,
        "by_question_type": metrics.by_question_type,
    }
