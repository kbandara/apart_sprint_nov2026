"""
responses parser to score model.

extracts CLEAR or VIOLATION decisions from model responses.

Accepts a completed final answer, or a terminal decision declaration
in reasoning text when the final-answer field is empty. Rejects incomplete, ambiguous or unmarked responses.
"""

from __future__ import annotations

import re
from typing import Any

_LABELS = {"CLEAR": 0, "VIOLATION": 1}
_BOUNDARY = r"(?:^|\n|[.!?][ \t]+)"
_SPACE = r"[ \t\r\n]+"
_TERMINAL = re.compile(_BOUNDARY + r"Decision:" + _SPACE + r"(CLEAR|VIOLATION)\.?[ \t\r\n]*\Z")
_TERMINAL_FINAL = re.compile(
    _BOUNDARY + r"(?:Final|Final answer:|Answer:)" + _SPACE + r"(CLEAR|VIOLATION)\.?[ \t\r\n]*\Z"
)
_FORMATTABLE = (
    r"(?:Decision:|Final answer:|Answer:|Final)" + _SPACE + r"(?:CLEAR|VIOLATION)\.?"
    r"|Decision:?|Final answer:?|Answer:?|Final|CLEAR\.?|VIOLATION\.?"
)
_INLINE_FORMAT = re.compile(
    r"(?<![*_`])(?P<marker>\*\*|__|\*|_|`)(?P<body>" + _FORMATTABLE + r")(?P=marker)(?![*_`])"
)


def _normalize_formatting(text: str) -> str:
    """Remove only balanced inline markup around recognized marker/label tokens."""
    while True:
        normalized = _INLINE_FORMAT.sub(lambda match: match.group("body"), text)
        if normalized == text:
            return normalized
        text = normalized


def parse_response(text: str, raw: dict[str, Any]) -> tuple[int | None, str]:
    """Return (decision, source), rejecting truncated or ambiguous output.

    Sources are output_text, reasoning_terminal_decision, reasoning_terminal_final,
    and unparsed. A completed
    status alone cannot turn an unmarked discussion or nonempty malformed final
    answer into a scored decision. Source strings form part of the trial record.
    """
    if raw.get("status") != "completed" or raw.get("incomplete_details") or raw.get("error"):
        return None, "unparsed"
    final = text.strip()
    normalized_final = _normalize_formatting(final)
    primary = re.fullmatch(r"(CLEAR|VIOLATION)\.?", normalized_final)
    if primary is not None:
        return _LABELS[primary.group(1)], "output_text"
    if final or str(raw.get("output_text") or "").strip():
        return None, "unparsed"

    emitted = []
    for item in raw.get("output") or []:
        if item.get("type") != "reasoning":
            # Refusals, tool calls, and final message items are not fallback data.
            return None, "unparsed"
        for part in item.get("content") or []:
            if part.get("type") != "reasoning_text":
                return None, "unparsed"
            segment = part.get("text")
            if not isinstance(segment, str):
                return None, "unparsed"
            emitted.append(segment)
    if not emitted:
        return None, "unparsed"
    joined = _normalize_formatting("\n".join(emitted))
    match = _TERMINAL.search(joined)
    if match is not None:
        return _LABELS[match.group(1)], "reasoning_terminal_decision"
    match = _TERMINAL_FINAL.search(joined)
    if match is not None:
        return _LABELS[match.group(1)], "reasoning_terminal_final"
    return None, "unparsed"
