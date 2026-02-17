"""Minimal fallback for environments where native orjson wheels are blocked.

This module intentionally implements only the subset used by the local app stack.
It mirrors the most common `orjson` API points with stdlib `json`.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Optional

OPT_INDENT_2 = 1 << 0
OPT_SORT_KEYS = 1 << 1
OPT_APPEND_NEWLINE = 1 << 2
OPT_NON_STR_KEYS = 1 << 3
OPT_SERIALIZE_NUMPY = 1 << 4
OPT_PASSTHROUGH_DATETIME = 1 << 5
OPT_PASSTHROUGH_SUBCLASS = 1 << 6
OPT_OMIT_MICROSECONDS = 1 << 7
OPT_NAIVE_UTC = 1 << 8
OPT_UTC_Z = 1 << 9


class JSONEncodeError(TypeError):
    pass


JSONDecodeError = json.JSONDecodeError


def dumps(
    obj: Any,
    default: Optional[Callable[[Any], Any]] = None,
    option: int = 0,
) -> bytes:
    kwargs = {}
    if option & OPT_INDENT_2:
        kwargs["indent"] = 2
    if option & OPT_SORT_KEYS:
        kwargs["sort_keys"] = True

    def _default(value: Any) -> Any:
        if default is not None:
            return default(value)
        if option & OPT_SERIALIZE_NUMPY:
            try:
                import numpy as np  # type: ignore

                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
            except Exception:
                pass
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(f"Type is not JSON serializable: {type(value)!r}")

    try:
        text = json.dumps(obj, default=_default, ensure_ascii=False, **kwargs)
    except TypeError as exc:
        raise JSONEncodeError(str(exc)) from exc

    if option & OPT_APPEND_NEWLINE:
        text += "\n"
    return text.encode("utf-8")


def loads(data: Any) -> Any:
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data).decode("utf-8")
    return json.loads(data)
