"""JSON helpers that preserve Polish characters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def dumps_json(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)
