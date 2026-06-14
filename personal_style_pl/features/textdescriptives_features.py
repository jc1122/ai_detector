"""Optional TextDescriptives features (entropy, dependency distance, POS proportions).

Readability indices are English-calibrated; treat as relative-only for Polish.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_CLEAR_ERROR = (
    "TextDescriptives is required for this feature set but is unavailable. "
    "Install with `.[rich]` (lexicalrichness, textdescriptives) on the 3.12 env, "
    "or pass enabled=False to skip."
)
# VERIFIED non-NaN columns with the pl_nask pipeline (audit 2026-06-14). entropy/perplexity are
# EXCLUDED: TextDescriptives' information_theory needs a spaCy lexeme-probability table, which does
# not exist for Polish -> NaN. Real perplexity/entropy come from perplexity_features.py (papuGaPT2).
_COLUMNS = (
    "dependency_distance_mean",
    "dependency_distance_std",
    "prop_adjacent_dependency_relation_mean",
    "prop_adjacent_dependency_relation_std",
    "pos_prop_NOUN",
    "pos_prop_VERB",
    "pos_prop_ADJ",
    "pos_prop_ADP",
    "syllables_per_token_mean",
    "syllables_per_token_std",
    "token_length_mean",
    "token_length_std",
)


def _load_pipe(lang: str):
    import spacy
    import textdescriptives  # noqa: F401  (registers factories)
    try:
        nlp = spacy.load("pl_nask")
    except Exception:
        nlp = spacy.blank("pl")
    # NOTE: no information_theory pipe (entropy/perplexity are NaN for Polish).
    for comp in ("textdescriptives/dependency_distance",
                 "textdescriptives/pos_proportions",
                 "textdescriptives/descriptive_stats"):
        try:
            nlp.add_pipe(comp)
        except Exception:
            pass
    return nlp


class TextDescriptivesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lang: str = "pl", enabled: bool = True):
        self.lang = lang
        self.enabled = enabled
        self._nlp = None
        self._names: list[str] = []

    def _ensure(self):
        if self._nlp is not None:
            return
        try:
            self._nlp = _load_pipe(self.lang)
        except Exception as exc:
            raise RuntimeError(f"{_CLEAR_ERROR} (cause: {type(exc).__name__}: {exc})") from exc

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        texts = list(X)
        if not self.enabled:
            self._names = []
            return np.empty((len(texts), 0), dtype=float)
        self._ensure()
        import textdescriptives as td
        rows = []
        for text in texts:
            doc = self._nlp(text or " ")
            df = td.extract_df(doc, include_text=False)
            row = {}
            for c in _COLUMNS:
                val = float(df[c].iloc[0]) if c in df.columns else 0.0
                row[c] = 0.0 if val != val else val  # NaN -> 0.0
            rows.append(row)
        self._names = list(_COLUMNS)
        return np.asarray([[r[c] for c in _COLUMNS] for r in rows], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._names or _COLUMNS, dtype=object)
