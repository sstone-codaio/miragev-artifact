from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Metrics:
    n: int
    accuracy: float
    pair_accuracy: Optional[float]
    by_module: Dict[str, Dict[str, float]]
    by_domain: Dict[str, Dict[str, float]]
    by_question_type: Dict[str, Dict[str, float]]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(pred_rows: List[Dict[str, Any]]) -> Metrics:
    n = len(pred_rows)
    correct = sum(1 for r in pred_rows if r.get("is_correct") is True)
    accuracy = _safe_div(correct, n)

    # Pair accuracy: per (pair_id, question_id) require both roles correct.
    grouped: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    for r in pred_rows:
        pair_id = r.get("pair_id")
        role = r.get("pair_role")
        qid = r.get("question_id") or "q"
        if pair_id and role:
            grouped[(pair_id, qid)][role] = bool(r.get("is_correct"))

    pair_scores: List[int] = []
    for (_pair, _qid), roles in grouped.items():
        # require at least A and B
        if "A" in roles and "B" in roles:
            pair_scores.append(1 if (roles["A"] and roles["B"]) else 0)

    pair_accuracy = _safe_div(sum(pair_scores), len(pair_scores)) if pair_scores else None

    def breakdown(key: str) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pred_rows:
            k = r.get(key) or "unknown"
            buckets[str(k)].append(r)
        out: Dict[str, Dict[str, float]] = {}
        for k, rows in buckets.items():
            cn = len(rows)
            cc = sum(1 for rr in rows if rr.get("is_correct") is True)
            out[k] = {"n": cn, "accuracy": _safe_div(cc, cn)}
        return out

    return Metrics(
        n=n,
        accuracy=accuracy,
        pair_accuracy=pair_accuracy,
        by_module=breakdown("module"),
        by_domain=breakdown("domain"),
        by_question_type=breakdown("question_type"),
    )
