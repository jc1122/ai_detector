"""Build a one-class personal StyleProfile from writing samples."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..config import WEAK_PROFILE_WARNING, STYLE_NOT_AUTHORSHIP_WARNING
from ..features.surface_features import SurfaceFeatureExtractor, SURFACE_FEATURE_NAMES
from ..io import SampleDoc
from ..textsplit import chunk_document

_SCALE_FLOOR = 1e-6
_NEAR_ZERO_VAR = 1e-9


@dataclass
class StyleProfile:
    profile_id: str
    created_at: str
    language: str
    feature_names: list[str]
    center: np.ndarray
    scale: np.ndarray
    robust_center: np.ndarray
    robust_scale: np.ndarray
    stable_mask: np.ndarray
    training_scores: list[float]
    training_sample_count: int
    training_chunk_count: int
    total_tokens: int
    genres: list[str]
    config: dict
    warnings: list[str] = field(default_factory=list)
    temperature: float = 1.0
    ngram_extractor: object | None = None  # fitted NgramFeatureExtractor, reused at scoring


def _robust_center_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    q75, q25 = np.percentile(matrix, [75, 25], axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > _SCALE_FLOOR, iqr, 1.0)
    return center, scale


def build_profile(
    docs: list[SampleDoc],
    *,
    use_stylometrix: bool = False,
    include_ngrams: bool = False,
    chunk_sentences: int = 8,
    min_chunk_tokens: int = 120,
) -> StyleProfile:
    chunks = []
    genres = sorted({d.genre for d in docs if d.genre})
    for doc in docs:
        chunks.extend(chunk_document(
            doc.text, doc_id=doc.doc_id,
            chunk_sentences=chunk_sentences, min_chunk_tokens=min_chunk_tokens))
    if not chunks:
        raise RuntimeError("No usable chunks could be built from the samples.")
    chunk_texts = [c.text for c in chunks]

    # Assemble feature blocks, then compute statistics ONCE over the full matrix so the
    # scorer can reproduce identical dimensions.
    feature_names = list(SURFACE_FEATURE_NAMES)
    blocks = [SurfaceFeatureExtractor().fit_transform(chunk_texts)]

    if use_stylometrix:
        from ..features.stylometrix_features import StyloMetrixFeatureExtractor
        sm_ext = StyloMetrixFeatureExtractor(enabled=True)
        blocks.append(sm_ext.fit_transform(chunk_texts))
        feature_names += list(sm_ext.get_feature_names_out())

    ngram_extractor = None
    if include_ngrams and len(chunks) >= 4:
        from ..features.ngram_features import NgramFeatureExtractor
        ngram_extractor = NgramFeatureExtractor(min_df=2).fit(chunk_texts)
        blocks.append(ngram_extractor.transform(chunk_texts))
        feature_names += list(ngram_extractor.get_feature_names_out())

    matrix = blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > _SCALE_FLOOR, std, 1.0)
    robust_center, robust_scale = _robust_center_scale(matrix)
    stable_mask = matrix.var(axis=0) > _NEAR_ZERO_VAR

    warnings = [STYLE_NOT_AUTHORSHIP_WARNING]
    total_tokens = sum(c.token_count for c in chunks)
    if len(chunks) < 10 or total_tokens < 5000:
        warnings.append(WEAK_PROFILE_WARNING)

    profile = StyleProfile(
        profile_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        language="pl",
        feature_names=feature_names,
        center=mean,
        scale=std,
        robust_center=robust_center,
        robust_scale=robust_scale,
        stable_mask=stable_mask,
        training_scores=[],
        training_sample_count=len(docs),
        training_chunk_count=len(chunks),
        total_tokens=total_tokens,
        genres=genres,
        config={
            "use_stylometrix": use_stylometrix,
            "include_ngrams": include_ngrams,
            "chunk_sentences": chunk_sentences,
            "min_chunk_tokens": min_chunk_tokens,
        },
        warnings=warnings,
        ngram_extractor=ngram_extractor,
    )
    from .calibration import finalize_profile
    finalize_profile(profile, matrix)
    return profile
