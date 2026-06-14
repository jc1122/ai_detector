"""AI-direction overlay: read interpretable metrics against known AI directionality.

Marker-based heuristic, NOT authorship proof. Abstains/flags low confidence on Polish/OOD,
where AI classifiers are unreliable (false positives are the dominant harm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heuristic_detector import detect_language, _extract_words
from .features.rich_metrics import rich_metrics_for_text
from .features.surface_features import surface_features_for_text


@dataclass
class Marker:
    feature: str
    direction: int  # +1: higher = more AI-leaning; -1: lower = more AI-leaning
    rationale: str
    suggestion: str


AI_MARKERS = (
    Marker("burstiness_coeff", -1,
           "AI text has low sentence-length variation (burstiness).",
           "Vary sentence length — mix short and long sentences."),
    Marker("sentence_len_cv", -1,
           "Uniform sentence length (low CV) reads AI-like.",
           "Increase sentence-length variation."),
    Marker("mattr", -1,
           "AI text tends to lower lexical diversity.",
           "Replace repeated words with varied vocabulary (do not invent facts)."),
    Marker("em_dash_per_1k", +1,
           "LLMs overuse em-dashes.",
           "Cut em-dashes toward your usual rate."),
    Marker("boilerplate_per_1k", +1,
           "Formulaic/boilerplate phrases are an AI tell.",
           "Delete or replace generic phrases (e.g. 'warto zauważyć')."),
    Marker("transition_per_1k", +1,
           "Over-signposted transitions read AI-like.",
           "Reduce or vary transition phrases."),
    Marker("repeated_4gram_ratio", +1,
           "Repeated higher-order n-grams over-appear in machine text.",
           "Rephrase repeated multi-word chunks."),
)

PERPLEXITY_MARKERS = (
    Marker("lm_perplexity", -1,
           "AI text is lower-perplexity (more predictable) than human.",
           "(diagnostic) very low perplexity leans AI."),
    Marker("sent_perplexity_cv", -1,
           "Low sentence-level perplexity variation is AI-like.",
           "(diagnostic) flat per-sentence perplexity leans AI."),
)

_NOT_PROOF = "Marker-based heuristic, not proof of authorship."
_PL_WARN = "Polish input: AI classifiers are unreliable here; treat markers as advisory only."


def _all_metrics(text: str, with_perplexity: bool) -> dict[str, float]:
    metrics = dict(surface_features_for_text(text))
    metrics.update(rich_metrics_for_text(text))
    if with_perplexity:
        from .features.perplexity_features import perplexity_features_for_text
        metrics.update(perplexity_features_for_text(text))
    return metrics


def ai_marker_report(text, profile=None, with_perplexity: bool = False) -> dict:
    lang = detect_language(text)
    tokens = len(_extract_words(text))
    metrics = _all_metrics(text, with_perplexity)
    markers = list(AI_MARKERS) + (list(PERPLEXITY_MARKERS) if with_perplexity else [])

    rows: list[dict] = []
    flags: list[int] = []
    feature_names = list(getattr(profile, "feature_names", []) or [])
    for mk in markers:
        if mk.feature not in metrics:
            continue
        value = float(metrics[mk.feature])
        row = {
            "feature": mk.feature,
            "value": round(value, 4),
            "ai_direction": "higher" if mk.direction > 0 else "lower",
            "rationale": mk.rationale,
        }
        if profile is not None and mk.feature in feature_names:
            idx = feature_names.index(mk.feature)
            base = float(profile.center[idx])
            scale = float(profile.scale[idx]) or 1.0
            z = (value - base) / scale
            leaning_toward_ai = mk.direction * z  # >1 sigma in the AI direction = flagged
            row["your_baseline"] = round(base, 4)
            row["z_vs_you"] = round(z, 2)
            is_flag = leaning_toward_ai > 1.0
            row["leaning"] = "AI-leaning" if is_flag else (
                "more-human-than-you" if leaning_toward_ai < -1.0 else "matches")
            row["suggestion"] = mk.suggestion if is_flag else None
            flags.append(1 if is_flag else 0)
        rows.append(row)

    # Transparent aggregate: % of evaluated markers that lean AI vs YOUR baseline.
    ai_leaning_score = round(100.0 * float(np.mean(flags))) if flags else None
    confidence = "low" if (lang == "pl" or tokens < 200 or profile is None) else "medium"
    warnings = [_NOT_PROOF]
    if lang == "pl":
        warnings.append(_PL_WARN)
    if profile is None:
        warnings.append("No personal profile supplied: showing raw marker values only "
                        "(absolute AI thresholds are unreliable; provide a profile for scoring).")
    return {
        "language": lang,
        "tokens": tokens,
        "ai_leaning_score": ai_leaning_score,
        "confidence": confidence,
        "ood_or_unreliable": lang == "pl",
        "markers": rows,
        "warnings": warnings,
    }
