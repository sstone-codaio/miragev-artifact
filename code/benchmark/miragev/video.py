from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class FrameSamplingConfig:
    num_frames: int = 8
    target_fps: Optional[float] = None  # if set, overrides num_frames with fps-based sampling
    seed: int = 0
    shuffle: bool = False
    single_frame: bool = False
    focus_critical_window: bool = False
    critical_window_boost: int = 0  # extra frames to add within critical window (if available)


def _try_decord(path: str):
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(path, ctx=cpu(0))
        return vr
    except Exception:
        return None


def _try_opencv(path: str):
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        return cap
    except Exception:
        return None


def _sample_indices_uniform(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if num_frames <= 1:
        return [min(total_frames - 1, total_frames // 2)]
    idx = np.linspace(0, total_frames - 1, num=num_frames)
    idx = np.round(idx).astype(int).tolist()
    # Ensure sorted and unique-ish while preserving length by nudging duplicates
    fixed: List[int] = []
    used = set()
    for i in idx:
        j = int(i)
        while j in used and j + 1 < total_frames:
            j += 1
        used.add(j)
        fixed.append(j)
    return fixed


def _sample_indices_fps(total_frames: int, video_fps: float, target_fps: float) -> List[int]:
    if total_frames <= 0:
        return []
    if video_fps <= 0 or target_fps <= 0:
        return _sample_indices_uniform(total_frames, min(8, total_frames))
    step = max(1, int(round(video_fps / target_fps)))
    return list(range(0, total_frames, step))


def _filter_to_window(indices: List[int], total_frames: int, video_fps: float, window: Tuple[float, float]) -> List[int]:
    start_s, end_s = window
    start = int(max(0, round(start_s * video_fps)))
    end = int(min(total_frames - 1, round(end_s * video_fps)))
    return [i for i in indices if start <= i <= end]


def extract_frames(
    video_path: str,
    cfg: FrameSamplingConfig,
    critical_window: Optional[Tuple[float, float]] = None,
) -> List[Image.Image]:
    """Extract frames from a local video file.

    Requires at least one of:
      - decord (recommended)
      - opencv-python-headless

    Returns list of PIL images (RGB).
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    rng = random.Random(cfg.seed)

    vr = _try_decord(video_path)
    if vr is not None:
        total = len(vr)
        fps = float(getattr(vr, "get_avg_fps", lambda: 0.0)() or 0.0)

        if cfg.target_fps is not None:
            indices = _sample_indices_fps(total, fps if fps > 0 else 30.0, cfg.target_fps)
        else:
            indices = _sample_indices_uniform(total, cfg.num_frames)

        if cfg.focus_critical_window and critical_window and cfg.critical_window_boost > 0 and fps > 0:
            # Add more frames inside critical window.
            window_indices = _filter_to_window(list(range(total)), total, fps, critical_window)
            if window_indices:
                extra = [rng.choice(window_indices) for _ in range(cfg.critical_window_boost)]
                indices = sorted(set(indices + extra))

        if cfg.single_frame:
            indices = [indices[len(indices) // 2]] if indices else [min(total - 1, total // 2)]

        if cfg.shuffle:
            rng.shuffle(indices)

        batch = vr.get_batch(indices).asnumpy()  # (N,H,W,3) RGB
        frames = [Image.fromarray(arr).convert("RGB") for arr in batch]
        return frames

    # Fallback to OpenCV
    cap = _try_opencv(video_path)
    if cap is None:
        raise RuntimeError(
            "Could not decode video. Install decord or opencv-python-headless. "
            "Example: pip install decord opencv-python-headless"
        )

    import cv2  # type: ignore
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

    if cfg.target_fps is not None:
        indices = _sample_indices_fps(total, fps if fps > 0 else 30.0, cfg.target_fps)
    else:
        indices = _sample_indices_uniform(total, cfg.num_frames)

    if cfg.focus_critical_window and critical_window and cfg.critical_window_boost > 0 and fps > 0:
        window_indices = _filter_to_window(list(range(total)), total, fps, critical_window)
        if window_indices:
            extra = [rng.choice(window_indices) for _ in range(cfg.critical_window_boost)]
            indices = sorted(set(indices + extra))

    if cfg.single_frame:
        indices = [indices[len(indices) // 2]] if indices else [min(total - 1, total // 2)]

    if cfg.shuffle:
        rng.shuffle(indices)

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb).convert("RGB"))
    cap.release()
    return frames
