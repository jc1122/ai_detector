"""Optional character/word n-gram features (topic-sensitive; off by default)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class NgramFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, min_df: int = 2):
        self.min_df = min_df
        self._char = None
        self._word = None

    def fit(self, X, y=None):
        texts = list(X)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=self.min_df, lowercase=True)
        self._word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     min_df=self.min_df, lowercase=True)
        self._char.fit(texts)
        self._word.fit(texts)
        return self

    def transform(self, X) -> np.ndarray:
        texts = list(X)
        char = self._char.transform(texts).toarray()
        word = self._word.transform(texts).toarray()
        return np.hstack([char, word])

    def get_feature_names_out(self, input_features=None):
        names = ([f"char::{n}" for n in self._char.get_feature_names_out()] +
                 [f"word::{n}" for n in self._word.get_feature_names_out()])
        return np.asarray(names, dtype=object)
