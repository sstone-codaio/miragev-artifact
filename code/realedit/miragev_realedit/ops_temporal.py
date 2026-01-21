from __future__ import annotations

from typing import List, Tuple
import numpy as np


def clamp_seg(seg: Tuple[int, int], n: int) -> Tuple[int, int]:
    s, e = seg
    s = max(0, min(s, n))
    e = max(0, min(e, n))
    if e < s:
        s, e = e, s
    return s, e


def time_jump(frames: List[np.ndarray], *, window: Tuple[int, int], jump_to: int) -> List[np.ndarray]:
    """Replace frames[window] with frames starting at jump_to.

    This creates an apparent discontinuity without pixel-level manipulation.
    Works best when the camera is static.
    """
    n = len(frames)
    s, e = clamp_seg(window, n)
    L = e - s
    if L <= 0:
        return frames.copy()
    jt = max(0, min(jump_to, n - L))
    out = frames.copy()
    out[s:e] = frames[jt:jt+L]
    return out


def swap_segments(frames: List[np.ndarray], *, seg1: Tuple[int, int], seg2: Tuple[int, int]) -> List[np.ndarray]:
    """Swap two segments in time.

    Use for TR-ORDER when two events occur in disjoint windows.
    """
    n = len(frames)
    a0, a1 = clamp_seg(seg1, n)
    b0, b1 = clamp_seg(seg2, n)
    if a1 <= a0 or b1 <= b0:
        return frames.copy()
    if (a0 < b0 < a1) or (b0 < a0 < b1):
        # overlapping segments; not supported
        return frames.copy()
    out = frames.copy()
    segA = out[a0:a1]
    segB = out[b0:b1]
    # Swap by rebuilding list
    if a0 < b0:
        return out[:a0] + segB + out[a1:b0] + segA + out[b1:]
    else:
        return out[:b0] + segA + out[b1:a0] + segB + out[a1:]


def splice(a: List[np.ndarray], b: List[np.ndarray], *, cut: int) -> List[np.ndarray]:
    """Take a[:cut] then b[cut:].

    Useful for cross-take occlusion splices and energy splices.
    """
    n = min(len(a), len(b))
    cut = max(0, min(cut, n))
    return a[:cut] + b[cut:n]


def loop_segment(frames: List[np.ndarray], *, seg: Tuple[int, int], times: int = 2) -> List[np.ndarray]:
    """Loop a segment to change event counts (⚠️ often creates artifacts)."""
    n = len(frames)
    s, e = clamp_seg(seg, n)
    if e <= s or times <= 1:
        return frames.copy()
    out = frames[:s] + frames[s:e] * times + frames[e:]
    return out
