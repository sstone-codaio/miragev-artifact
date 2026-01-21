from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", type=str, default="runs/*/metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.runs_glob))
    if not paths:
        raise SystemExit(f"No metrics.json files matched: {args.runs_glob}")

    rows: List[Dict[str, Any]] = []
    for mp in paths:
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        run_dir = os.path.dirname(mp)
        cfgp = os.path.join(run_dir, "run_config.json")
        cfg = {}
        if os.path.exists(cfgp):
            with open(cfgp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        rows.append({
            "run_dir": run_dir,
            "model_id": cfg.get("model_id"),
            "adapter": cfg.get("adapter"),
            "split": cfg.get("split"),
            "mode": cfg.get("mode"),
            "num_frames": cfg.get("num_frames"),
            "fps": cfg.get("fps"),
            "accuracy": m.get("accuracy"),
            "pair_accuracy": m.get("pair_accuracy"),
            "n": m.get("n"),
        })

    # Print as CSV-ish
    headers = ["run_dir","model_id","adapter","split","mode","num_frames","fps","accuracy","pair_accuracy","n"]
    print(",".join(headers))
    for r in rows:
        print(",".join([str(r.get(h, "")) for h in headers]))


if __name__ == "__main__":
    main()
