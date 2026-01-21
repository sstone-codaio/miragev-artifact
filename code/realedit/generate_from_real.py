from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from miragev_realedit.video_io import read_video, write_video, trim_to_same_length
from miragev_realedit.ops_temporal import time_jump, swap_segments, splice, loop_segment
from miragev_realedit.ops_roi import BBox, micro_teleport_roi, swap_two_rois, track_bbox_csrt
from miragev_realedit.recipes import load_recipe_yaml, validate_recipe


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=str, required=True, help="YAML recipe describing sources and edits")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--pairs_per_module", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap for reading frames (debug)")
    return ap.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def qa_row(
    *,
    example_id: str,
    pair_id: str,
    pair_role: str,
    question_type: str,
    prompt: str,
    options: Optional[List[str]],
    answer_index: int,
    video_path: str,
    module: str,
    critical_window_s: Tuple[float, float],
    generation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "example_id": example_id,
        "pair_id": pair_id,
        "pair_role": pair_role,
        "question_id": "q1",
        "question_type": question_type,
        "prompt": prompt,
        "options": options,
        "answer_index": int(answer_index),
        "video_path": video_path,
        "module": module,
        "domain": module,
        "critical_window": [float(critical_window_s[0]), float(critical_window_s[1])],
        "source_type": "real_edited",
        "generation": generation,
    }


def choose(rng: np.random.Generator, xs: List[Any]) -> Any:
    if not xs:
        raise ValueError("Empty source list")
    return xs[int(rng.integers(0, len(xs)))]


def seconds_to_frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    recipe = load_recipe_yaml(args.recipe)
    validate_recipe(recipe)

    out_root = args.out_root
    videos_dir = os.path.join(out_root, "videos")
    ann_dir = os.path.join(out_root, "annotations")
    pair_meta_dir = os.path.join(ann_dir, "pair_meta")
    ensure_dir(videos_dir)
    ensure_dir(pair_meta_dir)

    jsonl_path = os.path.join(ann_dir, "miragev.jsonl")
    if not os.path.exists(jsonl_path):
        ensure_dir(ann_dir)
        open(jsonl_path, "w", encoding="utf-8").close()

    # Cache decoded sources to avoid re-reading.
    cache: Dict[str, Any] = {}

    def get_video(path: str):
        if path not in cache:
            cache[path] = read_video(path, max_frames=args.max_frames)
        return cache[path]

    for mod in recipe["modules"]:
        module = mod["module"]
        kind = mod["kind"]
        q = mod["question"]
        question_type = q.get("type", "yesno")
        prompt = q["prompt"]
        options = q.get("options")
        # convention: yesno uses 1=YES, 0=NO
        ansA = int(q["answer_A"])
        ansB = int(q["answer_B"])
        critical_s = tuple(q.get("critical_window_s", [0.0, 1.0]))
        assert len(critical_s) == 2

        # Some edits need manual-ish parameters; recipe provides ranges (seconds or pixels).
        for i in tqdm(range(args.pairs_per_module), desc=f"Generating {module}"):
            pair_id = f"{module}_{i:04d}"

            outA = os.path.join(videos_dir, module, f"pair_{pair_id}_A.mp4")
            outB = os.path.join(videos_dir, module, f"pair_{pair_id}_B.mp4")
            ensure_dir(os.path.dirname(outA))

            gen_meta: Dict[str, Any] = {"module": module, "kind": kind, "recipe": mod.get("name", module), "seed": int(args.seed)}

            if kind == "intra_timejump":
                src = mod["source"]
                v = get_video(src)
                frames = v.frames
                fps = v.fps

                # sample time window for the jump
                win_s = mod.get("window_s", [1.5, 2.0])  # seconds range for start
                dur_s = mod.get("duration_s", [0.15, 0.25])  # seconds window length
                jump_ahead_s = mod.get("jump_ahead_s", [0.5, 1.2])

                s0 = float(rng.uniform(win_s[0], win_s[1]))
                d = float(rng.uniform(dur_s[0], dur_s[1]))
                j = float(rng.uniform(jump_ahead_s[0], jump_ahead_s[1]))

                s = seconds_to_frames(s0, fps)
                e = seconds_to_frames(s0 + d, fps)
                jt = seconds_to_frames(s0 + j, fps)

                framesA = frames
                framesB = time_jump(frames, window=(s, e), jump_to=jt)

                write_video(outA, framesA, fps=fps)
                write_video(outB, framesB, fps=fps)
                gen_meta.update({"source": src, "time_jump": {"window_frames": [s, e], "jump_to": jt}})

            elif kind == "intra_swap":
                src = mod["source"]
                v = get_video(src)
                frames = v.frames
                fps = v.fps

                seg1_s = mod.get("seg1_s", [1.0, 1.5])
                seg2_s = mod.get("seg2_s", [2.0, 2.5])
                seg_len_s = mod.get("seg_len_s", [0.2, 0.35])

                a0 = float(rng.uniform(seg1_s[0], seg1_s[1]))
                b0 = float(rng.uniform(seg2_s[0], seg2_s[1]))
                L = float(rng.uniform(seg_len_s[0], seg_len_s[1]))

                seg1 = (seconds_to_frames(a0, fps), seconds_to_frames(a0 + L, fps))
                seg2 = (seconds_to_frames(b0, fps), seconds_to_frames(b0 + L, fps))

                framesA = frames
                framesB = swap_segments(frames, seg1=seg1, seg2=seg2)

                write_video(outA, framesA, fps=fps)
                write_video(outB, framesB, fps=fps)
                gen_meta.update({"source": src, "swap_segments": {"seg1": seg1, "seg2": seg2}})

            elif kind == "cross_splice":
                srcA = choose(rng, mod["sources_A"])
                srcB = choose(rng, mod["sources_B"])
                vA = get_video(srcA)
                vB = get_video(srcB)

                # Require same resolution; if not, you should preprocess externally
                if (vA.width, vA.height) != (vB.width, vB.height):
                    raise ValueError(f"Resolution mismatch: {srcA} vs {srcB}")

                vA2, vB2 = trim_to_same_length(vA, vB)
                framesA0, framesB0 = vA2.frames, vB2.frames
                fps = vA2.fps

                # pick a cut time (e.g., during occlusion or right at impact)
                cut_s_range = mod.get("cut_s", [1.5, 2.5])
                cut_s = float(rng.uniform(cut_s_range[0], cut_s_range[1]))
                cut = seconds_to_frames(cut_s, fps)

                out_frames_A = framesA0  # A is pure take A
                out_frames_B = splice(framesA0, framesB0, cut=cut)

                write_video(outA, out_frames_A, fps=fps)
                write_video(outB, out_frames_B, fps=fps)
                gen_meta.update({"sources": {"A": srcA, "B": srcB}, "cut_frame": cut})

            elif kind == "roi_swap":
                src = mod["source"]
                v = get_video(src)
                frames = v.frames
                fps = v.fps

                # BBox definitions. Prefer providing init bboxes and letting tracker run.
                init1 = mod["bbox1"]  # [x,y,w,h]
                init2 = mod["bbox2"]

                # Optional tracking
                use_tracker = bool(mod.get("use_tracker", True))
                if use_tracker:
                    bxs1 = track_bbox_csrt(frames, BBox(*init1))
                    bxs2 = track_bbox_csrt(frames, BBox(*init2))
                else:
                    bxs1 = [BBox(*init1) for _ in frames]
                    bxs2 = [BBox(*init2) for _ in frames]

                # Swap range
                swap_s = mod.get("swap_s", [2.0, 3.0])
                swap_len_s = mod.get("swap_len_s", [0.6, 1.2])
                s0 = float(rng.uniform(swap_s[0], swap_s[1]))
                L = float(rng.uniform(swap_len_s[0], swap_len_s[1]))
                s = seconds_to_frames(s0, fps)
                e = seconds_to_frames(s0 + L, fps)

                # Apply per-frame swap using tracked bboxes
                framesB = [f.copy() for f in frames]
                for fi in range(max(0, s), min(len(framesB), e)):
                    # swap patches at that frame
                    framesB[fi:fi+1] = swap_two_rois(framesB[fi:fi+1], bbox1=bxs1[fi], bbox2=bxs2[fi], frame_range=(0, 1))

                write_video(outA, frames, fps=fps)
                write_video(outB, framesB, fps=fps)
                gen_meta.update({"source": src, "roi_swap": {"bbox1": init1, "bbox2": init2, "swap_frames": [s, e], "tracker": use_tracker}})

            elif kind == "pair_only":
                # Not an edit: simply pair two real takes (latent friction/mass).
                srcA = choose(rng, mod["sources_A"])
                srcB = choose(rng, mod["sources_B"])
                vA = get_video(srcA)
                vB = get_video(srcB)
                if (vA.width, vA.height) != (vB.width, vB.height):
                    raise ValueError(f"Resolution mismatch: {srcA} vs {srcB}")
                vA2, vB2 = trim_to_same_length(vA, vB)
                # Optional: choose a random subclip
                max_start_s = float(mod.get("max_start_s", 0.0))
                fps = vA2.fps
                start = 0
                if max_start_s > 0.0:
                    start = int(rng.integers(0, seconds_to_frames(max_start_s, fps) + 1))
                # Keep same length
                L = min(len(vA2.frames), len(vB2.frames)) - start
                framesA0 = vA2.frames[start:start+L]
                framesB0 = vB2.frames[start:start+L]
                write_video(outA, framesA0, fps=fps)
                write_video(outB, framesB0, fps=fps)
                gen_meta.update({"sources": {"A": srcA, "B": srcB}, "start_frame": start})

            else:
                raise ValueError(f"Unknown kind: {kind}")

            # Write per-pair meta
            write_json(os.path.join(pair_meta_dir, f"pair_{pair_id}.json"), gen_meta)

            # JSONL rows (flat schema compatible with MIRAGE-V evaluator)
            relA = os.path.relpath(outA, out_root)
            relB = os.path.relpath(outB, out_root)

            rows = [
                qa_row(
                    example_id=f"{pair_id}_A",
                    pair_id=pair_id,
                    pair_role="A",
                    question_type=question_type,
                    prompt=prompt,
                    options=options,
                    answer_index=ansA,
                    video_path=relA,
                    module=module,
                    critical_window_s=critical_s,
                    generation=gen_meta | {"variant": "A"},
                ),
                qa_row(
                    example_id=f"{pair_id}_B",
                    pair_id=pair_id,
                    pair_role="B",
                    question_type=question_type,
                    prompt=prompt,
                    options=options,
                    answer_index=ansB,
                    video_path=relB,
                    module=module,
                    critical_window_s=critical_s,
                    generation=gen_meta | {"variant": "B"},
                ),
            ]
            append_jsonl(jsonl_path, rows)

    print(f"Done. Wrote: {jsonl_path}")
    print(f"Videos under: {videos_dir}")


if __name__ == "__main__":
    main()
