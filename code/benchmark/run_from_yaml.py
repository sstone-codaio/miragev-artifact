from __future__ import annotations

import argparse
import subprocess
import sys
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Convert YAML to CLI arguments
    cli = ["python", "evaluate.py"]
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cli.append(flag)
        else:
            cli.extend([flag, str(v)])

    print("Running:", " ".join(cli))
    raise SystemExit(subprocess.call(cli))


if __name__ == "__main__":
    main()
