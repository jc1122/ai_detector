"""Length-robust, interpretable rich metrics (borrows LexicalRichness)."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from lexicalrichness import LexicalRichness
from sklearn.base import BaseEstimator, TransformerMixin

from heuristic_detector import _extract_words, _split_sentences, _fold_text
from ..config import TRANSITION_PHRASES, BOILERPLATE_PHRASES, HEDGE_PHRASES

RICH_METRIC_NAMES = (
    "mattr", "mtld", "hdd", "rttr", "cttr",
    "sentence_len_cv", "burstiness_coeff", "repeated_4gram_ratio",
    "em_dash_per_1k", "ai_phrase_per_1k", "boilerplate_per_1k",
    "transition_per_1k", "hedge_per_1k",
)

_EMDASH_RE = re.compile(r"[—–]")  # — –


def _safe(n: float, d: float) -> float:
    return n / d if d else 0.0


def _burstiness(lengths: list[int]) -> float:
    if len(lengths) < 2:
        return 0.0
    arr = np.asarray(lengths, dtype=float)
    mu, sd = float(arr.mean()), float(arr.std())
    return (sd - mu) / (sd + mu) if (sd + mu) > 0 else 0.0


def _repeated_ngram_ratio(words: list[str], n: int = 4) -> float:
    if len(words) < n + 1:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return _safe(sum(c for c in grams.values() if c > 1), sum(grams.values()))


def _count_phrases(folded: str, phrases) -> int:
    return sum(folded.count(p) for p in phrases)


def _lexical_diversity(text: str) -> dict[str, float]:
    try:
        lex = LexicalRichness(text)
        w = lex.words
        if w < 2:
            return {"mattr": 0.0, "mtld": 0.0, "hdd": 0.0, "rttr": 0.0, "cttr": 0.0}
        return {
            "mattr": float(lex.mattr(window_size=min(50, w))),
            "mtld": float(lex.mtld()) if w >= 10 else 0.0,
            "hdd": float(lex.hdd(draws=min(42, w))) if w >= 42 else 0.0,
            "rttr": float(lex.rttr),
            "cttr": float(lex.cttr),
        }
    except Exception:
        return {"mattr": 0.0, "mtld": 0.0, "hdd": 0.0, "rttr": 0.0, "cttr": 0.0}


def rich_metrics_for_text(text: str) -> dict[str, float]:
    text = text or ""
    words = _extract_words(text)
    folded_words = [_fold_text(w) for w in words]
    sents = _split_sentences(text)
    sent_lens = [len(_extract_words(s)) for s in sents] or [0]
    folded = _fold_text(text)
    ntok = max(len(words), 1)

    feats = _lexical_diversity(text)
    feats.update({
        "sentence_len_cv": _safe(float(np.std(sent_lens)), float(np.mean(sent_lens))),
        "burstiness_coeff": _burstiness(sent_lens),
        "repeated_4gram_ratio": _repeated_ngram_ratio(folded_words, 4),
        "em_dash_per_1k": _safe(len(_EMDASH_RE.findall(text)), ntok) * 1000.0,
        "ai_phrase_per_1k": _safe(
            _count_phrases(folded, TRANSITION_PHRASES + BOILERPLATE_PHRASES), ntok) * 1000.0,
        "boilerplate_per_1k": _safe(_count_phrases(folded, BOILERPLATE_PHRASES), ntok) * 1000.0,
        "transition_per_1k": _safe(_count_phrases(folded, TRANSITION_PHRASES), ntok) * 1000.0,
        "hedge_per_1k": _safe(_count_phrases(folded, HEDGE_PHRASES), ntok) * 1000.0,
    })
    return feats


class RichMetricsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in X:
            m = rich_metrics_for_text(text if isinstance(text, str) else "")
            rows.append([m[name] for name in RICH_METRIC_NAMES])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(RICH_METRIC_NAMES, dtype=object)
