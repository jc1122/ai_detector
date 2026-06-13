"""Serialize ScoreResult to dict/JSON."""

from __future__ import annotations

from dataclasses import asdict

from .similarity import ScoreResult


def score_result_to_dict(result: ScoreResult) -> dict:
    return asdict(result)
