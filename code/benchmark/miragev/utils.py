from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def retry(
    fn,
    *,
    tries: int = 3,
    base_sleep_s: float = 1.0,
    jitter_s: float = 0.25,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Small retry helper for flaky API calls."""
    import random as _random

    last = None
    for i in range(tries):
        try:
            return fn()
        except exceptions as e:
            last = e
            sleep = base_sleep_s * (2**i) + _random.random() * jitter_s
            time.sleep(sleep)
    raise last  # type: ignore[misc]
