"""Deterministic, Polish-aware surface style features (sklearn transformer)."""

from __future__ import annotations

import re
import statistics
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from heuristic_detector import (
    _extract_words,
    _split_sentences,
    _split_paragraphs,
    _fold_text,
)
from ..config import (
    TRANSITION_PHRASES,
    BOILERPLATE_PHRASES,
    HEDGE_PHRASES,
    FIRST_PERSON_SINGULAR,
    FIRST_PERSON_PLURAL,
    CLAUSE_MARKERS,
)

SURFACE_FEATURE_NAMES = (
    "char_count", "token_count", "sentence_count", "paragraph_count",
    "avg_sentence_len_tokens", "median_sentence_len_tokens", "std_sentence_len_tokens",
    "sentence_len_cv", "min_sentence_len_tokens", "max_sentence_len_tokens",
    "avg_paragraph_len_sentences", "avg_token_len_chars", "type_token_ratio",
    "hapax_ratio", "punctuation_density", "comma_per_sentence", "semicolon_per_sentence",
    "colon_per_sentence", "dash_per_sentence", "question_mark_count",
    "exclamation_mark_count", "parenthesis_density", "quote_density", "digit_density",
    "uppercase_ratio", "newline_density", "bullet_like_line_ratio",
    "repeated_bigram_ratio", "repeated_trigram_ratio", "top_token_repetition_ratio",
    "first_person_singular_count", "first_person_plural_count", "hedge_count",
    "transition_phrase_count", "generic_boilerplate_count", "average_clause_marker_count",
)

_BULLET_RE = re.compile(r"^\s*([-–—•*]|\d+[.)])\s+")

# Punctuation set using explicit unicode codepoints to avoid source encoding issues.
_PUNCT_CHARS = frozenset(
    ".,;:!?()-"
    + "–—"          # en-dash, em-dash
    + "\"'‘’‚‛“”„‟"  # quote variants
    + "«»"           # guillemets
    + "…"                 # ellipsis
    + "[]{}"
)

# Regex pattern for quote characters (for quote_density).
_QUOTE_RE = re.compile(
    "[\"\'‘’‚‛“”„‟«»]"
)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _count_phrases(folded_text: str, phrases) -> int:
    return sum(folded_text.count(p) for p in phrases)


def _ngram_repeat_ratio(words: list[str], size: int) -> float:
    if len(words) < size + 1:
        return 0.0
    grams = Counter(tuple(words[i : i + size]) for i in range(len(words) - size + 1))
    repeated = sum(c for c in grams.values() if c > 1)
    return _safe_div(repeated, sum(grams.values()))


def surface_features_for_text(text: str) -> dict[str, float]:
    words = _extract_words(text)
    folded_words = [_fold_text(w) for w in words]
    sentences = _split_sentences(text)
    paragraphs = _split_paragraphs(text)
    sent_lens = [len(_extract_words(s)) for s in sentences] or [0]
    folded_text = _fold_text(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    token_counts = Counter(folded_words)

    mean_len = statistics.fmean(sent_lens)
    std_len = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    sample = folded_words[:200]

    feats = {
        "char_count": float(len(text)),
        "token_count": float(len(words)),
        "sentence_count": float(len(sentences)),
        "paragraph_count": float(len(paragraphs)),
        "avg_sentence_len_tokens": mean_len,
        "median_sentence_len_tokens": float(statistics.median(sent_lens)),
        "std_sentence_len_tokens": std_len,
        "sentence_len_cv": _safe_div(std_len, mean_len),
        "min_sentence_len_tokens": float(min(sent_lens)),
        "max_sentence_len_tokens": float(max(sent_lens)),
        "avg_paragraph_len_sentences": _safe_div(len(sentences), len(paragraphs)),
        "avg_token_len_chars": _safe_div(sum(len(w) for w in words), len(words)),
        "type_token_ratio": _safe_div(len(set(sample)), len(sample)),
        "hapax_ratio": _safe_div(
            sum(1 for c in token_counts.values() if c == 1), len(token_counts)),
        "punctuation_density": _safe_div(
            sum(1 for ch in text if ch in _PUNCT_CHARS), len(text)),
        "comma_per_sentence": _safe_div(text.count(","), len(sentences)),
        "semicolon_per_sentence": _safe_div(text.count(";"), len(sentences)),
        "colon_per_sentence": _safe_div(text.count(":"), len(sentences)),
        "dash_per_sentence": _safe_div(
            len(re.findall("[-–—]", text)), len(sentences)),
        "question_mark_count": float(text.count("?")),
        "exclamation_mark_count": float(text.count("!")),
        "parenthesis_density": _safe_div(text.count("(") + text.count(")"), len(text)),
        "quote_density": _safe_div(len(_QUOTE_RE.findall(text)), len(text)),
        "digit_density": _safe_div(sum(ch.isdigit() for ch in text), len(text)),
        "uppercase_ratio": _safe_div(
            sum(1 for ch in text if ch.isupper()),
            sum(1 for ch in text if ch.isalpha())),
        "newline_density": _safe_div(text.count("\n"), len(text)),
        "bullet_like_line_ratio": _safe_div(
            sum(1 for ln in lines if _BULLET_RE.match(ln)), len(lines)),
        "repeated_bigram_ratio": _ngram_repeat_ratio(folded_words, 2),
        "repeated_trigram_ratio": _ngram_repeat_ratio(folded_words, 3),
        "top_token_repetition_ratio": _safe_div(
            max(token_counts.values(), default=0), len(words)),
        "first_person_singular_count": float(
            sum(token_counts[w] for w in FIRST_PERSON_SINGULAR)),
        "first_person_plural_count": float(
            sum(token_counts[w] for w in FIRST_PERSON_PLURAL)),
        "hedge_count": float(_count_phrases(folded_text, HEDGE_PHRASES)),
        "transition_phrase_count": float(_count_phrases(folded_text, TRANSITION_PHRASES)),
        "generic_boilerplate_count": float(
            _count_phrases(folded_text, BOILERPLATE_PHRASES)),
        "average_clause_marker_count": _safe_div(
            sum(token_counts[w] for w in CLAUSE_MARKERS), max(len(sentences), 1)),
    }
    return feats


class SurfaceFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible deterministic surface feature extractor."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in X:
            feats = surface_features_for_text(text if isinstance(text, str) else "")
            rows.append([feats[name] for name in SURFACE_FEATURE_NAMES])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(SURFACE_FEATURE_NAMES, dtype=object)
