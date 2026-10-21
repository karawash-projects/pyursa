"""Structured, agent-readable errors.

Every error carries a stable ``code``, a human message, machine-readable
``details`` and a list of ``suggestions``. Each suggestion may include a
``fix`` snippet an agent can paste back into its code. Errors are raised
while the plan is being *built*, before any engine runs, whenever possible.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, Iterable


class PyursaError(Exception):
    code = "PYURSA_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        suggestions: list[dict[str, str]] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": _jsonable(self.details),
            "suggestions": self.suggestions,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        lines = [f"[{self.code}] {self.message}"]
        for s in self.suggestions:
            fix = f"  ->  {s['fix']}" if s.get("fix") else ""
            lines.append(f"  suggestion: {s['description']}{fix}")
        return "\n".join(lines)


class ColumnNotFoundError(PyursaError):
    code = "COLUMN_NOT_FOUND"


class TypeMismatchError(PyursaError):
    code = "TYPE_MISMATCH"


class InvalidOperationError(PyursaError):
    code = "INVALID_OPERATION"


class DuplicateColumnError(PyursaError):
    code = "DUPLICATE_COLUMN"


class UnsupportedTypeError(PyursaError):
    code = "UNSUPPORTED_TYPE"


class BackendError(PyursaError):
    code = "BACKEND_ERROR"


class UnknownBackendError(PyursaError):
    code = "UNKNOWN_BACKEND"


class HistoryError(PyursaError):
    code = "HISTORY_ERROR"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _is_subsequence(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(ch in it for ch in short)


def column_not_found(name: str, available: Iterable[str], where: str) -> ColumnNotFoundError:
    available = list(available)
    lower = {a.lower(): a for a in available}
    candidates: list[str] = []
    if name.lower() in lower:
        candidates.append(lower[name.lower()])
    for c in difflib.get_close_matches(name, available, n=3, cutoff=0.6):
        if c not in candidates:
            candidates.append(c)
    for c in available:  # abbreviations: qty <-> quantity, amt <-> amount
        short, long_ = sorted((c.lower(), name.lower()), key=len)
        if c not in candidates and len(short) >= 2 and short[0] == long_[0] and _is_subsequence(short, long_):
            candidates.append(c)
    candidates = candidates[:3]
    suggestions = [
        {"description": f"Did you mean '{c}'?", "fix": f'col("{c}")'} for c in candidates
    ]
    if not suggestions:
        suggestions.append(
            {"description": "Check the schema with frame.schema or frame.describe_for_llm()."}
        )
    return ColumnNotFoundError(
        f"Column '{name}' not found in {where}.",
        details={"column": name, "available_columns": available, "location": where},
        suggestions=suggestions,
    )
