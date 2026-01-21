from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from miragev.data import load_miragev
from miragev.evaluator import run_evaluation
from miragev.models.registry import create_adapter
from miragev.utils import dump_json, ensure_dir, now_iso, set_seed, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="MIRAGE-V evaluation harness")

    p.add_argument("--dataset", type=str, required=True,
                   help="HF dataset name (ORG/MIRAGE-V) OR local JSON/JSONL file path")
    p.add_argument("--split", type=str, default="test", help="HF split (ignored for local JSON/JSONL)")

    # Model adapter
    p.add_argument("--adapter", type=str, default="hf", choices=["hf", "openai", "gemini_video", "gemini_frames"],
                   help="Which model adapter to use")

    # HF
    p.add_argument("--model_name_or_path", type=str, default=None,
                   help="Hugging Face model path or repo id")
    p.add_argument("--hf_device_map", type=str, default="auto")
    p.add_argument("--hf_dtype", type=str, default="auto")
    p.add_argument("--hf_max_new_tokens", type=int, default=16)
    p.add_argument("--hf_temperature", type=float, default=0.0)
    p.add_argument("--hf_top_p", type=float, default=1.0)
    p.add_argument("--hf_trust_remote_code", action="store_true", default=True)
    p.add_argument("--hf_image_token", type=str, default="<image>")
    p.add_argument("--hf_chat_template", type=str, default="auto", choices=["auto", "none"])

    # OpenAI
    p.add_argument("--openai_model", type=str, default="gpt-5.2")
    p.add_argument("--openai_max_output_tokens", type=int, default=128)

    # Gemini
    p.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    p.add_argument("--cache_dir", type=str, default=".cache/gemini")

    # Video / sampling
    p.add_argument("--mode", type=str, default="full_video",
                   choices=["full_video", "text_only", "single_frame", "shuffled_frames"])
    p.add_argument("--num_frames", type=int, default=8,
                   help="Uniformly sample this many frames (ignored if --fps set)")
    p.add_argument("--fps", type=float, default=None,
                   help="Target fps to sample (overrides --num_frames)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--focus_critical_window", action="store_true", default=False)
    p.add_argument("--critical_window_boost", type=int, default=0,
                   help="Extra frames to sample inside critical window (if present)")

    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--video_root", type=str, default=None) 
    p.add_argument("--out_dir", type=str, default="runs/latest")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    ds = load_miragev(args.dataset, split=args.split)
    rows: List[Dict[str, Any]] = [ds[i] for i in range(len(ds))]

    # Build adapter
    if args.adapter == "hf":
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required for --adapter hf")
        adapter = create_adapter(
            "hf",
            model_name_or_path=args.model_name_or_path,
            device_map=args.hf_device_map,
            dtype=args.hf_dtype,
            max_new_tokens=args.hf_max_new_tokens,
            temperature=args.hf_temperature,
            top_p=args.hf_top_p,
            trust_remote_code=args.hf_trust_remote_code,
            image_token=args.hf_image_token,
            chat_template=args.hf_chat_template,
        )
        model_id = args.model_name_or_path
    elif args.adapter == "openai":
        adapter = create_adapter(
            "openai",
            model=args.openai_model,
            max_output_tokens=args.openai_max_output_tokens,
        )
        model_id = args.openai_model
    elif args.adapter == "gemini_video":
        adapter = create_adapter(
            "gemini_video",
            model=args.gemini_model,
            cache_dir=args.cache_dir,
        )
        model_id = args.gemini_model
    elif args.adapter == "gemini_frames":
        adapter = create_adapter(
            "gemini_frames",
            model=args.gemini_model,
        )
        model_id = args.gemini_model
    else:
        raise ValueError(f"Unknown adapter: {args.adapter}")

    preds, metrics = run_evaluation(
        rows,
        adapter,
        mode=args.mode,
        num_frames=args.num_frames,
        target_fps=args.fps,
        seed=args.seed,
        focus_critical_window=args.focus_critical_window,
        critical_window_boost=args.critical_window_boost,
        max_examples=args.max_examples,
        video_root=args.video_root,
    )

    # Write outputs
    write_jsonl(os.path.join(args.out_dir, "predictions.jsonl"), preds)
    dump_json(os.path.join(args.out_dir, "metrics.json"), metrics)
    dump_json(os.path.join(args.out_dir, "run_config.json"), vars(args) | {"ts": now_iso(), "model_id": model_id})

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
