"""One-class style similarity scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (
    BLEND_COSINE, BLEND_Z, LABEL_THRESHOLDS, MIN_CANDIDATE_TOKENS,
    STYLE_NOT_AUTHORSHIP_WARNING,
)
from ..features.surface_features import SurfaceFeatureExtractor, SURFACE_FEATURE_NAMES
from ..textsplit import chunk_document
from heuristic_detector import _extract_words
from .calibration import chunk_distance


@dataclass
class ScoreResult:
    style_match_score: float
    label: str
    confidence: str
    warnings: list[str]
    summary: str
    top_matches: list[dict]
    top_mismatches: list[dict]
    chunk_scores: list[dict] = field(default_factory=list)


def _label(score: float) -> str:
    if score >= LABEL_THRESHOLDS["close_to_my_style"]:
        return "close_to_my_style"
    if score >= LABEL_THRESHOLDS["mixed"]:
        return "mixed"
    return "far_from_my_style"


def _confidence(profile, candidate_tokens: int) -> str:
    chunks = profile.training_chunk_count
    if chunks >= 30 and candidate_tokens >= 300:
        return "high"
    if chunks >= 15 and candidate_tokens >= 200:
        return "medium"
    return "low"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _features_for_chunks(profile, texts: list[str]) -> np.ndarray:
    """Reproduce the EXACT feature pipeline the profile was built with, so dimensions
    match profile.center/scale. Surface + StyloMetrix are stateless; the n-gram
    vectorizer is the fitted one persisted on the profile."""
    blocks = [SurfaceFeatureExtractor().fit_transform(texts)]
    if profile.config.get("use_stylometrix"):
        from ..features.stylometrix_features import StyloMetrixFeatureExtractor
        blocks.append(StyloMetrixFeatureExtractor(enabled=True).fit_transform(texts))
    if profile.ngram_extractor is not None:
        blocks.append(profile.ngram_extractor.transform(texts))
    return blocks[0] if len(blocks) == 1 else np.hstack(blocks)


def score_text(profile, text: str) -> ScoreResult:
    warnings = [STYLE_NOT_AUTHORSHIP_WARNING]
    candidate_tokens = len(_extract_words(text))
    if candidate_tokens < MIN_CANDIDATE_TOKENS:
        return ScoreResult(
            style_match_score=0.0, label="insufficient_text",
            confidence="low",
            warnings=["Very short candidate: score may be unstable."] + warnings,
            summary="Text too short for a reliable style estimate.",
            top_matches=[], top_mismatches=[], chunk_scores=[])

    chunks = chunk_document(
        text, doc_id="candidate",
        chunk_sentences=profile.config.get("chunk_sentences", 8),
        min_chunk_tokens=profile.config.get("min_chunk_tokens", 120))
    chunks_texts = [c.text for c in chunks] if chunks else [text]

    matrix = _features_for_chunks(profile, chunks_texts)
    chunk_scores = []
    per_chunk = []
    for idx, row in enumerate(matrix):
        distance = chunk_distance(profile, row)
        z_component = 100.0 * float(np.exp(-distance / profile.temperature))
        # Cosine similarity of the candidate chunk to the profile centroid (raw feature
        # space). All surface features are non-negative, so this term is high for any
        # well-formed Polish text; it is intentionally the minor (0.3) blend term that
        # rewards overall shape agreement while the robust-z term does the discrimination.
        cosine_component = 100.0 * max(0.0, _cosine(row, profile.center))
        score = BLEND_Z * z_component + BLEND_COSINE * cosine_component
        score = max(0.0, min(100.0, score))
        per_chunk.append(score)
        chunk_scores.append({"chunk_id": idx, "score": round(score), "label": _label(score)})

    arr = np.asarray(per_chunk)
    aggregate = float(0.5 * np.median(arr) + 0.5 * np.percentile(arr, 25))
    label = _label(aggregate)
    confidence = _confidence(profile, candidate_tokens)
    if profile.training_chunk_count < 10:
        confidence = "low"
        warnings.append("Small style profile: add more samples.")

    top_matches, top_mismatches = _feature_diffs(profile, matrix.mean(axis=0))
    summary = (f"Style match {round(aggregate)}/100 ({label}); confidence {confidence}. "
               f"Based on {len(chunks_texts)} candidate chunk(s).")
    return ScoreResult(
        style_match_score=round(aggregate, 1), label=label, confidence=confidence,
        warnings=warnings, summary=summary, top_matches=top_matches,
        top_mismatches=top_mismatches, chunk_scores=chunk_scores)


def _feature_diffs(profile, candidate_mean: np.ndarray):
    rows = []
    for i, name in enumerate(profile.feature_names):
        if not profile.stable_mask[i]:
            continue
        std = profile.scale[i]
        z = (candidate_mean[i] - profile.center[i]) / std
        rows.append((abs(z), {
            "feature": name,
            "candidate_value": round(float(candidate_mean[i]), 3),
            "profile_mean": round(float(profile.center[i]), 3),
            "profile_std": round(float(std), 3),
            "z": round(float(z), 2),
        }))
    rows.sort(key=lambda r: r[0])
    matches = [dict(r[1], effect="matches") for r in rows[:5]]
    mismatches = []
    for _, d in sorted(rows, key=lambda r: -r[0])[:5]:
        d = dict(d)
        d["effect"] = "higher_than_usual" if d["z"] > 0 else "lower_than_usual"
        mismatches.append(d)
    return matches, mismatches
