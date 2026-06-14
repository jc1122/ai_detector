"""Optional Polish-LM perplexity + curvature (papuGaPT2).

VERIFIED (audit 2026-06-14): the model loads in ~21 s and computes per-sentence perplexity. Two
findings baked in below: (1) per-sentence perplexity has heavy OUTLIERS from PDF-extraction noise,
so aggregation is ROBUST (median + IQR), not mean/std; (2) on the Polish chemistry domain perplexity
did NOT cleanly separate human from AI-assisted text (median PPL human≈50, AI-assisted≈54) — so this
is an ADVISORY signal, not a verdict. Polarity: lower perplexity / lower per-sentence dispersion lean AI.

Gated: requires transformers + the papuGaPT2 model.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from heuristic_detector import _split_sentences

_MODEL_NAME = "dkleczek/papuGaPT2"
_CLEAR_ERROR = (
    "Polish-LM perplexity features require `transformers` + the papuGaPT2 model. "
    "Fetch it once (see scripts/setup_style_env.sh), or omit --with-perplexity."
)
_CACHE: dict = {}

PERPLEXITY_FEATURE_NAMES = (
    "lm_perplexity",         # robust: MEDIAN per-sentence perplexity
    "lm_logprob_mean",       # mean per-token log-prob over the whole text
    "sent_perplexity_cv",    # robust dispersion: IQR / median of per-sentence perplexity
    "fastdetect_curvature",  # std of per-sentence log-perplexity (single-model proxy)
)


def _load_model(name: str):
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if name not in _CACHE:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        _CACHE[name] = (tok, model)
    return _CACHE[name]


def _nll(text: str, tok, model) -> float | None:
    import torch
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids
    if ids.shape[1] < 2:
        return None
    with torch.no_grad():
        return float(model(ids, labels=ids).loss)  # mean NLL per token


def perplexity_features_for_text(text: str, model_name: str = _MODEL_NAME) -> dict[str, float]:
    try:
        tok, model = _load_model(model_name)
    except Exception as exc:
        raise RuntimeError(f"{_CLEAR_ERROR} (cause: {type(exc).__name__}: {exc})") from exc

    doc_nll = _nll(text, tok, model)
    sents = _split_sentences(text) or [text]
    ppls = [math.exp(min(nll, 20.0)) for s in sents if (nll := _nll(s, tok, model)) is not None]
    if not ppls:
        return {k: 0.0 for k in PERPLEXITY_FEATURE_NAMES}
    arr = np.asarray(ppls, dtype=float)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    robust_cv = float((q75 - q25) / median) if median > 0 else 0.0
    return {
        "lm_perplexity": median,
        "lm_logprob_mean": float(-doc_nll) if doc_nll is not None else float(-math.log(median + 1e-9)),
        "sent_perplexity_cv": robust_cv,
        "fastdetect_curvature": float(np.log(arr + 1e-9).std()),
    }


class PerplexityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, enabled: bool = True, model_name: str = _MODEL_NAME):
        self.enabled = enabled
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = [[perplexity_features_for_text(t if isinstance(t, str) else " ", self.model_name)[n]
                 for n in PERPLEXITY_FEATURE_NAMES] for t in X]
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(PERPLEXITY_FEATURE_NAMES, dtype=object)
