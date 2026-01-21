from __future__ import annotations

from typing import List, Optional, Tuple

SYSTEM_INSTRUCTION = (
    "You are a strict benchmark model evaluator for MIRAGE-V, a video physics-checking benchmark. "
    "You must answer based only on the visual evidence in the provided video frames or video. "
    "Do not add explanations unless explicitly requested. "
    "If uncertain, make your best guess."
)


def _letters(n: int) -> List[str]:
    # Up to 26 options (A-Z)
    return [chr(ord("A") + i) for i in range(n)]


def build_prompt(
    question_type: str,
    question: str,
    options: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Return (instructions, user_prompt)."""
    qt = (question_type or "mcq").lower()
    if qt == "yesno":
        instructions = SYSTEM_INSTRUCTION + " Reply with ONLY 'YES' or 'NO'."
        user = f"Question: {question}\nAnswer:"
        return instructions, user

    # default: MCQ
    opts = options or []
    letters = _letters(len(opts))
    formatted = "\n".join([f"{letters[i]}. {opts[i]}" for i in range(len(opts))])

    instructions = SYSTEM_INSTRUCTION + " Reply with ONLY the letter of the correct option (A, B, C, ...)."
    user = f"Question: {question}\nOptions:\n{formatted}\nAnswer:"
    return instructions, user
