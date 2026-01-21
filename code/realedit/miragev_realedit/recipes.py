from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


# A "recipe" describes how to create minimal pairs for one module using real video(s).
# This repo keeps recipes simple and explicit because editing requires human supervision.


def _require(d: Dict[str, Any], k: str):
    if k not in d:
        raise ValueError(f"Recipe missing required field '{k}': {d}")
    return d[k]


def load_recipe_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_recipe(recipe: Dict[str, Any]) -> None:
    _require(recipe, "modules")
    if not isinstance(recipe["modules"], list):
        raise ValueError("recipe['modules'] must be a list")
    for m in recipe["modules"]:
        _require(m, "module")
        _require(m, "kind")  # intra_timejump | intra_swap | cross_splice | roi_swap | pair_only
        _require(m, "question")
        # Sources
        if m["kind"] in ("intra_timejump", "intra_swap", "roi_swap"):
            _require(m, "source")
        elif m["kind"] in ("cross_splice", "pair_only"):
            _require(m, "sources_A")
            _require(m, "sources_B")
