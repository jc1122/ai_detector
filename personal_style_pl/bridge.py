"""Bridge to the existing heuristic_detector AI-likeness scorer."""

from __future__ import annotations

import heuristic_detector


def attach_heuristics(payload: dict, text: str) -> dict:
    """Add an `ai_likeness` block from heuristic_detector alongside style similarity.
    This is style/AI-likeness signal, NOT proof of authorship."""
    try:
        analysis = heuristic_detector.analyze_text(text)
        payload = dict(payload)
        payload["ai_likeness"] = {
            "ai_probability": float(analysis["ai_probability"]),
            "language": analysis["language"],
            "note": "Heuristic AI-likeness, not proof of authorship.",
        }
    except RuntimeError as exc:
        payload = dict(payload)
        payload["ai_likeness"] = {"error": str(exc)}
    return payload
