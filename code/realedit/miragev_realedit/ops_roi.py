from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


def _clip_bbox(b: BBox, W: int, H: int) -> BBox:
    x = max(0, min(b.x, W - 1))
    y = max(0, min(b.y, H - 1))
    w = max(1, min(b.w, W - x))
    h = max(1, min(b.h, H - y))
    return BBox(x, y, w, h)


def _inpaint_background(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # OpenCV inpaint expects 8-bit 1-channel mask where non-zero pixels are inpainted
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(frame_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def micro_teleport_roi(
    frames: List[np.ndarray],
    *,
    bbox: BBox,
    frame_range: Tuple[int, int],
    dx: int,
    dy: int,
) -> List[np.ndarray]:
    """Move a rectangular ROI by (dx,dy) for a short range of frames.

    ⚠️ Requires a relatively simple background; inpainting may leave artifacts.
    Works best when the ROI is small and motion is mild.
    """
    out = [f.copy() for f in frames]
    n = len(out)
    s = max(0, min(frame_range[0], n))
    e = max(0, min(frame_range[1], n))
    if e <= s:
        return out

    H, W = out[0].shape[:2]
    b = _clip_bbox(bbox, W, H)
    for i in range(s, e):
        fr = out[i]
        patch = fr[b.y:b.y+b.h, b.x:b.x+b.w].copy()

        # mask old location for inpainting
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[b.y:b.y+b.h, b.x:b.x+b.w] = 255
        fr_bg = _inpaint_background(fr, mask)

        # paste at new location
        nx = int(np.clip(b.x + dx, 0, W - b.w))
        ny = int(np.clip(b.y + dy, 0, H - b.h))
        fr_bg[ny:ny+b.h, nx:nx+b.w] = patch
        out[i] = fr_bg

    return out


def swap_two_rois(
    frames: List[np.ndarray],
    *,
    bbox1: BBox,
    bbox2: BBox,
    frame_range: Tuple[int, int],
) -> List[np.ndarray]:
    """Swap two ROIs for a frame range.

    This is a crude TM-ID editor. It works best when:
      - objects are similar size and appearance
      - camera is static
      - bboxes closely follow objects (manual or tracked)
    """
    out = [f.copy() for f in frames]
    n = len(out)
    s = max(0, min(frame_range[0], n))
    e = max(0, min(frame_range[1], n))
    if e <= s:
        return out

    H, W = out[0].shape[:2]
    b1 = _clip_bbox(bbox1, W, H)
    b2 = _clip_bbox(bbox2, W, H)

    for i in range(s, e):
        fr = out[i]
        p1 = fr[b1.y:b1.y+b1.h, b1.x:b1.x+b1.w].copy()
        p2 = fr[b2.y:b2.y+b2.h, b2.x:b2.x+b2.w].copy()
        fr[b1.y:b1.y+b1.h, b1.x:b1.x+b1.w] = p2
        fr[b2.y:b2.y+b2.h, b2.x:b2.x+b2.w] = p1
        out[i] = fr

    return out


def track_bbox_csrt(frames: List[np.ndarray], init_bbox: BBox) -> List[BBox]:
    """Track a bbox over time using OpenCV CSRT tracker.

    Returns a bbox per frame. If tracking fails, it repeats the last bbox.

    Note: OpenCV trackers can be sensitive. For best results:
      - good lighting
      - moderate motion
      - minimal occlusion
    """
    if not frames:
        return []
    H, W = frames[0].shape[:2]
    b0 = _clip_bbox(init_bbox, W, H)

    tracker = cv2.TrackerCSRT_create()
    first_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    ok = tracker.init(first_bgr, (b0.x, b0.y, b0.w, b0.h))
    bboxes = [b0]
    last = b0
    for i in range(1, len(frames)):
        fr_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        ok, bb = tracker.update(fr_bgr)
        if ok:
            x, y, w, h = [int(v) for v in bb]
            last = _clip_bbox(BBox(x, y, w, h), W, H)
        bboxes.append(last)
    return bboxes
