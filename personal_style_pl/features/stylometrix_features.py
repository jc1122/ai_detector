"""Optional StyloMetrix Polish feature extractor (failable)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_CLEAR_ERROR = (
    "StyloMetrix is required for this feature set but is unavailable. "
    "Run scripts/setup_style_env.sh (Python 3.12, numpy<2, pl_nask model), "
    "or pass enabled=False / --no-stylometrix to use surface features only."
)


def _import_stylo_metrix():
    import stylo_metrix as sm  # noqa: F401
    return sm


class StyloMetrixFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lang: str = "pl", enabled: bool = True):
        self.lang = lang
        self.enabled = enabled
        self._feature_names: list[str] = []
        self._stylo = None

    def _ensure(self):
        if self._stylo is not None:
            return
        try:
            sm = _import_stylo_metrix()
            self._stylo = sm.StyloMetrix(self.lang)
        except Exception as exc:  # ImportError, OSError (missing model), etc.
            raise RuntimeError(f"{_CLEAR_ERROR} (cause: {type(exc).__name__}: {exc})") from exc

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        texts = list(X)
        if not self.enabled:
            self._feature_names = []
            return np.empty((len(texts), 0), dtype=float)
        self._ensure()
        frame = self._stylo.transform(texts)
        cols = [c for c in frame.columns if c != "text"]
        self._feature_names = list(cols)
        return frame[cols].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._feature_names, dtype=object)
