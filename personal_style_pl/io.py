"""Load writing samples (dir or CSV) and write artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleDoc:
    doc_id: str
    text: str
    source: str | None = None
    date: str | None = None
    genre: str | None = None


def load_samples_from_dir(samples_dir: str | Path) -> list[SampleDoc]:
    directory = Path(samples_dir)
    if not directory.is_dir():
        raise RuntimeError(f"Samples directory not found: {directory}")
    docs: list[SampleDoc] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(SampleDoc(doc_id=path.stem, text=text))
    if not docs:
        raise RuntimeError(f"No .txt samples found in {directory}")
    return docs


def load_samples_from_csv(csv_path: str | Path, *, text_col: str = "text") -> list[SampleDoc]:
    path = Path(csv_path)
    if not path.is_file():
        raise RuntimeError(f"CSV not found: {path}")
    docs: list[SampleDoc] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise RuntimeError(f"CSV missing required text column '{text_col}'")
        for index, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            docs.append(SampleDoc(
                doc_id=row.get("source") or f"row{index}",
                text=text,
                source=row.get("source"),
                date=row.get("date"),
                genre=row.get("genre"),
            ))
    if not docs:
        raise RuntimeError(f"No rows with non-empty '{text_col}' in {path}")
    return docs


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
