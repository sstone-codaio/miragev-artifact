from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import imageio.v2 as imageio


@dataclass
class VideoData:
    frames: List[np.ndarray]   # list of HxWx3 uint8 RGB
    fps: float
    width: int
    height: int


def read_video(path: str, *, max_frames: Optional[int] = None) -> VideoData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: List[np.ndarray] = []
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        i += 1
        if max_frames is not None and i >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")

    h, w = frames[0].shape[:2]
    return VideoData(frames=frames, fps=fps, width=w, height=h)


def write_video(path: str, frames: List[np.ndarray], fps: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for fr in frames:
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            writer.append_data(fr)
    finally:
        writer.close()


def trim_to_same_length(a: VideoData, b: VideoData) -> Tuple[VideoData, VideoData]:
    n = min(len(a.frames), len(b.frames))
    a2 = VideoData(frames=a.frames[:n], fps=a.fps, width=a.width, height=a.height)
    b2 = VideoData(frames=b.frames[:n], fps=b.fps, width=b.width, height=b.height)
    return a2, b2
