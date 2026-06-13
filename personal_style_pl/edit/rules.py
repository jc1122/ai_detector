"""Conservative, deterministic text rules. Never invent facts."""

from __future__ import annotations

import re
from dataclasses import dataclass

_URL_RE = re.compile(r"https?://\S+")


@dataclass
class EditSuggestion:
    issue: str
    reason: str
    suggestion: str
    before: str | None = None
    after: str | None = None
    severity: str = "info"


def normalize_whitespace(text: str) -> str:
    # Protect URLs, collapse intra-line spaces, cap blank lines at one.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
