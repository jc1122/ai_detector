# Rich metrics v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an interpretable, length-normalized, OOD-aware "rich metrics" layer + an AI-direction overlay that flags literature-backed AI-leaning signals and feeds `suggest-edits`, borrowing proven libraries.

**Architecture:** New modules in `personal_style_pl` reusing the 3.12 StyloMetrix/spaCy env: `features/rich_metrics.py` (LexicalRichness + burstiness + n-gram + length-normalized densities), `features/textdescriptives_features.py` (spaCy TextDescriptives, gated), `features/perplexity_features.py` (papuGaPT2 + Fast-DetectGPT curvature, gated), and `ai_markers.py` (the overlay). CLI: `ai-markers` command + `suggest-edits`/detector integration.

**Tech Stack:** Python 3.12, numpy<2, LexicalRichness, TextDescriptives (spaCy v3), papuGaPT2 (optional), reusing `heuristic_detector` + `personal_style_pl`.

**Spec:** `docs/superpowers/specs/2026-06-14-rich-metrics-v2-design.md`

---

## Conventions
- TDD: failing test → run (fail) → implement → run (pass) → commit. Run with `.venv/bin/python -m pytest <path> -v`.
- Tests in `tests/personal_style/`. Commit messages: Conventional Commits, body ending with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Reuse: `from heuristic_detector import _extract_words, _split_sentences, _fold_text, detect_language`; `from personal_style_pl.config import TRANSITION_PHRASES, BOILERPLATE_PHRASES, HEDGE_PHRASES`.
- Branch: create `feature/rich-metrics-v2` off `main` before Task 1.

---

## Task 0: Deps, setup script, CI, branch

**Files:** `pyproject.toml`, `scripts/setup_style_env.sh`, `.github/workflows/ci.yml`

- [ ] **Step 1: Branch + install the rich libs (verified compatible: no spacy/numpy change)**
```bash
cd /home/jakub/projects/ai_detector
git checkout -b feature/rich-metrics-v2
~/.local/bin/uv pip install --python .venv/bin/python lexicalrichness textdescriptives "numpy<2"
.venv/bin/python -c "import lexicalrichness, textdescriptives; print('rich libs OK')"
```
Expected: prints `rich libs OK`; `.venv/bin/python -m pytest -q` still green (124).

- [ ] **Step 2: pyproject extras** — in `[project.optional-dependencies]` add:
```toml
rich = ["lexicalrichness", "textdescriptives"]
rich-perplexity = []  # uses existing torch+transformers; papuGaPT2 fetched by setup script
```
- [ ] **Step 3: setup script** — append to `scripts/setup_style_env.sh` before the final `echo`:
```bash
# Rich-metrics v2 extras (interpretable AI-leaning signals).
uv pip install --python .venv/bin/python -e ".[rich]" "numpy<2"
# Polish LM for perplexity/curvature features (~500 MB; required for --with-perplexity).
# VERIFIED: loads in ~21s and computes per-sentence perplexity on the 3.12 env.
.venv/bin/python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('dkleczek/papuGaPT2'); AutoModelForCausalLM.from_pretrained('dkleczek/papuGaPT2'); print('papuGaPT2 fetched')"
```
- [ ] **Step 4: CI** — in the `style-test` job's install step, change to:
```yaml
          python -m pip install -e ".[style]" lexicalrichness "numpy<2"
```
Only `lexicalrichness` in light CI (covers `rich_metrics`/`ai_markers` tests). `textdescriptives`
and perplexity tests are **model-gated** (skip without a real spaCy model / papuGaPT2), so they run
only in the full local env — do NOT add `textdescriptives` to the light job (it would pull a
model-less spaCy and make those tests meaningless).
- [ ] **Step 5: Commit**
```bash
git add pyproject.toml scripts/setup_style_env.sh .github/workflows/ci.yml
git commit -m "chore: add rich-metrics extras (lexicalrichness, textdescriptives) and CI wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: `features/rich_metrics.py`

**Files:** Create `personal_style_pl/features/rich_metrics.py`; Test `tests/personal_style/test_rich_metrics.py`

- [ ] **Step 1: Failing test** — `tests/personal_style/test_rich_metrics.py`:
```python
import numpy as np
from personal_style_pl.features.rich_metrics import (
    rich_metrics_for_text, RichMetricsExtractor, RICH_METRIC_NAMES,
)

PLAIN = ("Wczoraj poszedłem do sklepu. Kupiłem chleb i mleko. Wróciłem do domu. "
         "Zrobiłem herbatę i usiadłem przy oknie. ") * 3
DASHY = ("Sztuczna inteligencja — co warto wiedzieć — odgrywa rolę. "
         "Warto zauważyć, że to istotne — naprawdę istotne — dla wszystkich. ") * 3


def test_names_stable_and_complete():
    assert list(RichMetricsExtractor().fit([PLAIN]).get_feature_names_out()) == list(RICH_METRIC_NAMES)
    for req in ("mattr", "mtld", "burstiness_coeff", "em_dash_per_1k", "repeated_4gram_ratio"):
        assert req in RICH_METRIC_NAMES


def test_densities_length_normalized_and_emdash_detected():
    plain = rich_metrics_for_text(PLAIN)
    dashy = rich_metrics_for_text(DASHY)
    assert dashy["em_dash_per_1k"] > plain["em_dash_per_1k"]   # em-dash tell
    assert dashy["boilerplate_per_1k"] >= 0.0
    assert 0.0 <= plain["mattr"] <= 1.0


def test_empty_text_safe():
    X = RichMetricsExtractor().fit_transform([""])
    assert X.shape == (1, len(RICH_METRIC_NAMES))
    assert not np.isnan(X).any()
```
- [ ] **Step 2: Run → FAIL.** `.venv/bin/python -m pytest tests/personal_style/test_rich_metrics.py -v`

- [ ] **Step 3: Implement** — `personal_style_pl/features/rich_metrics.py`:
```python
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
```
- [ ] **Step 4: Run → PASS.** Then `.venv/bin/python -m pytest tests/personal_style -q` (all green).
- [ ] **Step 5: Commit**
```bash
git add personal_style_pl/features/rich_metrics.py tests/personal_style/test_rich_metrics.py
git commit -m "feat: rich_metrics (MATTR/MTLD/HD-D, burstiness, n-gram, length-normalized densities)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1b: Wire rich (+ optional perplexity) into the profile baseline

**Why:** the AI-direction overlay scores marker features against `profile.center/scale`. Those
features (`mattr`, `burstiness_coeff`, `*_per_1k`, …) must therefore live IN the profile. Mirror how
StyloMetrix/n-gram are wired. Rich metrics are cheap → on by default; perplexity is heavy → opt-in.

**Files:** Modify `personal_style_pl/profile/build_profile.py`, `personal_style_pl/profile/similarity.py`, `personal_style_pl/cli.py`; Test `tests/personal_style/test_profile_rich.py`

- [ ] **Step 1: Failing test** — `tests/personal_style/test_profile_rich.py`:
```python
from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text
from personal_style_pl.features.rich_metrics import RICH_METRIC_NAMES


def _docs():
    base = ("Wczoraj poszedłem do sklepu. Kupiłem chleb. Wróciłem do domu. "
            "Zrobiłem herbatę i usiadłem przy oknie spokojnie. ") * 4
    return [SampleDoc(doc_id=f"d{i}", text=base) for i in range(5)]


def test_profile_includes_rich_by_default_and_scores():
    p = build_profile(_docs(), use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)
    assert p.config["include_rich"] is True
    for name in ("mattr", "burstiness_coeff", "em_dash_per_1k", "boilerplate_per_1k"):
        assert name in p.feature_names
    # scorer must reproduce identical dimensions (no broadcast/hstack error)
    r = score_text(p, " ".join(d.text for d in _docs()[:1]))
    assert 0 <= r.style_match_score <= 100


def test_rich_can_be_disabled():
    p = build_profile(_docs(), use_stylometrix=False, include_rich=False,
                      chunk_sentences=2, min_chunk_tokens=15)
    assert p.config["include_rich"] is False
    assert "mattr" not in p.feature_names
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Edit `build_profile.build_profile`** — add params `include_rich: bool = True,
  include_perplexity: bool = False` to the signature. After the existing n-gram block (before the
  `matrix = blocks[0] if ...` line) insert:
```python
    if include_rich:
        from ..features.rich_metrics import RichMetricsExtractor, RICH_METRIC_NAMES
        blocks.append(RichMetricsExtractor().fit_transform(chunk_texts))
        feature_names += list(RICH_METRIC_NAMES)

    if include_perplexity:
        from ..features.perplexity_features import PerplexityExtractor, PERPLEXITY_FEATURE_NAMES
        blocks.append(PerplexityExtractor().fit_transform(chunk_texts))
        feature_names += list(PERPLEXITY_FEATURE_NAMES)
```
  and add to the `config={...}` dict: `"include_rich": include_rich, "include_perplexity": include_perplexity,`.

- [ ] **Step 4: Edit `similarity._features_for_chunks`** — after the n-gram block, before `return`,
  append (order MUST match build: surface → stylometrix → ngram → rich → perplexity):
```python
    if profile.config.get("include_rich"):
        from ..features.rich_metrics import RichMetricsExtractor
        blocks.append(RichMetricsExtractor().fit_transform(texts))
    if profile.config.get("include_perplexity"):
        from ..features.perplexity_features import PerplexityExtractor
        blocks.append(PerplexityExtractor().fit_transform(texts))
```

- [ ] **Step 5: Edit `cli.py` `build-profile`** — add `--no-rich` (sets `include_rich=False`) and
  `--with-perplexity` (sets `include_perplexity=True`) arguments; pass them into `build_profile(...)`.

- [ ] **Step 6: Run → PASS.** Then `.venv/bin/python -m pytest tests/personal_style -q` (all green;
  existing v1 profiles built without these flags still score because the scorer gates on
  `profile.config.get(...)`).

- [ ] **Step 7: Commit**
```bash
git add personal_style_pl/profile/build_profile.py personal_style_pl/profile/similarity.py personal_style_pl/cli.py tests/personal_style/test_profile_rich.py
git commit -m "feat: include rich (default) + perplexity (opt-in) metrics in StyleProfile baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `features/textdescriptives_features.py` (gated, spaCy)

**Files:** Create `personal_style_pl/features/textdescriptives_features.py`; Test `tests/personal_style/test_textdescriptives.py`

- [ ] **Step 1: Failing test** (importorskip — runs in our env, skips if dep absent):
```python
import numpy as np
import pytest


def test_disabled_returns_empty():
    from personal_style_pl.features.textdescriptives_features import TextDescriptivesExtractor
    ext = TextDescriptivesExtractor(enabled=False)
    X = ext.fit_transform(["Ala ma kota."])
    assert X.shape == (1, 0)


def test_missing_dependency_raises_clear_error(monkeypatch):
    import personal_style_pl.features.textdescriptives_features as m
    monkeypatch.setattr(m, "_load_pipe", lambda lang: (_ for _ in ()).throw(ImportError("no td")))
    ext = m.TextDescriptivesExtractor(enabled=True)
    with pytest.raises(RuntimeError, match="TextDescriptives"):
        ext.fit_transform(["Ala ma kota."])


def test_real_extraction():
    pytest.importorskip("textdescriptives")
    import spacy
    if not spacy.util.is_package("pl_nask"):
        pytest.skip("pl_nask model not installed (model-gated; runs only in full env)")
    from personal_style_pl.features.textdescriptives_features import TextDescriptivesExtractor
    ext = TextDescriptivesExtractor(enabled=True)
    X = ext.fit_transform(["Wczoraj poszedłem do urzędu i czekałem w kolejce dość długo, bo "
                           "formularz trzeba było poprawić aż dwa razy."])
    assert X.shape[0] == 1 and X.shape[1] == len(ext.get_feature_names_out())
    assert not np.isnan(X).any()           # NaN -> 0.0 guard works
    feats = dict(zip(ext.get_feature_names_out(), X[0]))
    assert feats["dependency_distance_mean"] > 0   # real parser ran (pl_nask)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `personal_style_pl/features/textdescriptives_features.py`:
```python
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
```
*Note:* the `_COLUMNS` above are **verified present and numeric** with `pl_nask` (audit 2026-06-14).
Do NOT add `entropy`/`perplexity` (NaN for Polish). NaN→0.0 is handled in `transform`.
- [ ] **Step 4: Run → PASS** (real test runs in `.venv`). `.venv/bin/python -m pytest tests/personal_style -q`.
- [ ] **Step 5: Commit**
```bash
git add personal_style_pl/features/textdescriptives_features.py tests/personal_style/test_textdescriptives.py
git commit -m "feat: optional TextDescriptives features (entropy, dependency distance, POS)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `ai_markers.py` — the AI-direction overlay

**Files:** Create `personal_style_pl/ai_markers.py`; Test `tests/personal_style/test_ai_markers.py`

- [ ] **Step 1: Failing test**:
```python
from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.ai_markers import ai_marker_report, AI_MARKERS


def _profile():
    base = ("Wczoraj poszedłem do sklepu. Kupiłem chleb. Wróciłem do domu. "
            "Zrobiłem herbatę i usiadłem przy oknie spokojnie. ") * 4
    return build_profile([SampleDoc(doc_id=f"d{i}", text=base) for i in range(4)],
                         use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)


def test_report_structure_and_pl_abstain():
    rep = ai_marker_report("Tekst po polsku. " * 30)
    assert "markers" in rep and "warnings" in rep
    assert rep["language"] == "pl"
    assert rep["ood_or_unreliable"] is True
    assert any("not proof of authorship" in w.lower() for w in rep["warnings"])
    assert any("polish" in w.lower() for w in rep["warnings"])


def test_markers_cover_known_signals():
    feats = {m.feature for m in AI_MARKERS}
    for req in ("burstiness_coeff", "em_dash_per_1k", "boilerplate_per_1k", "repeated_4gram_ratio"):
        assert req in feats


def test_overlay_with_profile_flags_dashy_boilerplate():
    profile = _profile()
    dashy = ("Warto zauważyć, że to istotne — naprawdę — dla wszystkich. "
             "Należy podkreślić kluczowe znaczenie — bez wątpienia. ") * 5
    rep = ai_marker_report(dashy, profile=profile)
    assert isinstance(rep["ai_leaning_score"], (int, float))
    flagged = {r["feature"] for r in rep["markers"] if r.get("leaning") == "AI-leaning"}
    assert "boilerplate_per_1k" in flagged or "em_dash_per_1k" in flagged
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `personal_style_pl/ai_markers.py`:
```python
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
```
- [ ] **Step 4: Run → PASS.** Then full `tests/personal_style -q`.
- [ ] **Step 5: Commit**
```bash
git add personal_style_pl/ai_markers.py tests/personal_style/test_ai_markers.py
git commit -m "feat: AI-direction overlay (interpretable, abstains on Polish/OOD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: CLI `ai-markers` + `suggest-edits` integration

**Files:** Modify `personal_style_pl/cli.py`, `personal_style_pl/edit/style_editor.py`; Test `tests/personal_style/test_ai_markers_cli.py`

- [ ] **Step 1: Failing test**:
```python
import json
from personal_style_pl.cli import main


def test_ai_markers_cli_json(tmp_path, capsys):
    draft = tmp_path / "d.txt"
    draft.write_text(("Warto zauważyć, że to istotne — naprawdę. " * 20), encoding="utf-8")
    rc = main(["ai-markers", "--text-file", str(draft), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"language", "markers", "warnings", "ood_or_unreliable"} <= set(payload)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — add handler + parser to `cli.py`:
```python
def _cmd_ai_markers(args) -> int:
    import joblib
    from .ai_markers import ai_marker_report
    from .utils.json import dumps_json
    text = Path(args.text_file).read_text(encoding="utf-8")
    profile = joblib.load(args.profile) if args.profile else None
    report = ai_marker_report(text, profile=profile, with_perplexity=args.with_perplexity)
    if args.json:
        print(dumps_json(report, indent=2))
    else:
        print(f"language={report['language']} ai_leaning_score={report['ai_leaning_score']} "
              f"confidence={report['confidence']}")
        for r in report["markers"]:
            print(f"- {r['feature']}={r['value']} ({r.get('leaning','-')})")
        for w in report["warnings"]:
            print(f"  ! {w}")
    return 0
```
In `build_parser()` before `return parser`:
```python
    am = sub.add_parser("ai-markers", help="Interpretable AI-leaning marker report.")
    am.add_argument("--text-file", required=True)
    am.add_argument("--profile")
    am.add_argument("--with-perplexity", action="store_true")
    am.add_argument("--json", action="store_true")
    am.set_defaults(func=_cmd_ai_markers)
```
- [ ] **Step 4:** Extend `suggest-edits` — in `edit/style_editor.py` `suggestions_to_markdown`, after the existing suggestions append an "AI-leaning markers" section:
```python
    from ..ai_markers import ai_marker_report
    overlay = ai_marker_report(text, profile=profile)
    lines.append("")
    lines.append("## AI-leaning markers (advisory)")
    if overlay["ai_leaning_score"] is not None:
        lines.append(f"- Overall AI-leaning vs your style: {overlay['ai_leaning_score']}/100")
    for r in overlay["markers"]:
        if r.get("leaning") == "AI-leaning" and r.get("suggestion"):
            lines.append(f"- `{r['feature']}` ({r['rationale']}) → {r['suggestion']}")
    for w in overlay["warnings"]:
        lines.append(f"- _{w}_")
```
- [ ] **Step 5: Run → PASS** (`test_ai_markers_cli.py` + existing `test_edit.py`). Verify `.venv/bin/python -m personal_style_pl.cli ai-markers --help`.
- [ ] **Step 6: Commit**
```bash
git add personal_style_pl/cli.py personal_style_pl/edit/style_editor.py tests/personal_style/test_ai_markers_cli.py
git commit -m "feat: ai-markers CLI and suggest-edits AI-leaning section

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `features/perplexity_features.py` (gated, papuGaPT2 + Fast-DetectGPT curvature)

**Files:** Create `personal_style_pl/features/perplexity_features.py`; Test `tests/personal_style/test_perplexity.py`

- [ ] **Step 1: Failing test** (importorskip — skips unless papuGaPT2 present):
```python
import pytest


def test_missing_model_raises_clear_error(monkeypatch):
    import personal_style_pl.features.perplexity_features as m
    monkeypatch.setattr(m, "_load_model", lambda name: (_ for _ in ()).throw(ImportError("no model")))
    with pytest.raises(RuntimeError, match="perplexity"):
        m.perplexity_features_for_text("Ala ma kota i psa.")


def test_real_perplexity():
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if try_to_load_from_cache("dkleczek/papuGaPT2", "config.json") is None:
        pytest.skip("papuGaPT2 not cached (model-gated; avoids network download in tests)")
    from personal_style_pl.features.perplexity_features import (
        perplexity_features_for_text, PerplexityExtractor, PERPLEXITY_FEATURE_NAMES)
    feats = perplexity_features_for_text(
        "Wczoraj poszedłem do urzędu. Czekałem w kolejce dość długo i nudno.")
    assert feats["lm_perplexity"] > 0
    assert set(PERPLEXITY_FEATURE_NAMES) <= set(feats)
    X = PerplexityExtractor().fit_transform(["Ala ma kota.", "Drugie zdanie tutaj jest tutaj."])
    assert X.shape == (2, len(PERPLEXITY_FEATURE_NAMES))
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `personal_style_pl/features/perplexity_features.py`:
```python
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
```
*Note:* this is a pragmatic single-model curvature proxy. A faithful Fast-DetectGPT implementation
(sampling alternative tokens from the same model and measuring conditional-probability curvature) can
replace `fastdetect_curvature` later; the interface stays the same.
- [ ] **Step 4: Run → PASS** (downloads papuGaPT2 on first run if fetched; else skips). Commit.
```bash
git add personal_style_pl/features/perplexity_features.py tests/personal_style/test_perplexity.py
git commit -m "feat: optional papuGaPT2 perplexity + curvature features (gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `ai-detector-heuristic --rich` integration

**Files:** Modify `heuristic_detector.py`; Test `tests/test_heuristic_rich.py`

- [ ] **Step 1: Failing test** (`tests/test_heuristic_rich.py`, importorskip on personal_style_pl deps):
```python
import json
import subprocess
import sys
import pytest


def test_rich_flag_adds_block(tmp_path):
    pytest.importorskip("lexicalrichness")
    f = tmp_path / "t.txt"
    f.write_text("Warto zauważyć, że to istotne — naprawdę. " * 20, encoding="utf-8")
    out = subprocess.run([sys.executable, "heuristic_detector.py", "--text-file", str(f),
                          "--json", "--rich"], capture_output=True, text=True)
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert "rich_metrics" in payload["experts"]["heuristic"]
    assert "ai_leaning" in payload["experts"]["heuristic"]
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — add a `--rich` argparse flag in `heuristic_detector.parse_args`, and in `build_payload`, when `rich` is set, attach (lazy import so the lean detector keeps zero deps):
```python
    if rich:
        try:
            from personal_style_pl.features.rich_metrics import rich_metrics_for_text
            from personal_style_pl.ai_markers import ai_marker_report
            expert_payload["rich_metrics"] = rich_metrics_for_text(text)
            expert_payload["ai_leaning"] = ai_marker_report(text)
        except Exception as exc:  # personal_style_pl deps not installed
            expert_payload["rich_metrics"] = {"error": str(exc)}
```
Thread `rich` through `main()`/`build_payload`. Keep all existing keys unchanged.
- [ ] **Step 4: Run → PASS.** Also confirm `python heuristic_detector.py --text "x ..." --json` (no `--rich`) still works with **no** personal_style_pl deps (lean path unchanged).
- [ ] **Step 5: Commit**
```bash
git add heuristic_detector.py tests/test_heuristic_rich.py
git commit -m "feat: ai-detector-heuristic --rich attaches rich metrics + AI-leaning overlay

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Hybrid `train-supervised --features rich`

**Files:** Modify `personal_style_pl/models/supervised.py`, `personal_style_pl/cli.py`; Test extend `tests/personal_style/test_supervised.py`

- [ ] **Step 1: Failing test** — add:
```python
def test_supervised_rich_features(tmp_path):
    import csv, joblib
    from personal_style_pl.cli import main
    p = tmp_path / "c.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["text", "label", "source"])
        for i in range(4):
            w.writerow([f"Mój zwięzły tekst osobisty numer {i}. Krótko i jasno.", "mine", f"m{i}"])
            w.writerow([f"Warto zauważyć — naprawdę — że kluczowe znaczenie ma rozwój {i}.", "other", f"o{i}"])
    out = tmp_path / "m.joblib"
    rc = main(["train-supervised", "--csv", str(p), "--features", "rich", "--output", str(out)])
    assert rc == 0 and out.exists()
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — in `supervised.py`, add a `feature_set` param; when `"rich"`, build the matrix by hstacking `SurfaceFeatureExtractor` + `RichMetricsExtractor`; store which feature set was used on the model. Add `--features {surface,rich}` (default `surface`) to the `train-supervised` parser and pass it through. `predict_proba_mine` rebuilds the same matrix.
- [ ] **Step 4: Run → PASS.** Commit.
```bash
git add personal_style_pl/models/supervised.py personal_style_pl/cli.py tests/personal_style/test_supervised.py
git commit -m "feat: supervised --features rich (surface + rich metrics hybrid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: README, AGENTS, acceptance

**Files:** `README.md`, `personal_style_pl/AGENTS.md`

- [ ] **Step 1:** README — append a "Rich metrics & AI-leaning markers (v2)" subsection under §12: what it measures, the `ai-markers` command, `--rich` detector flag, the **abstain-on-Polish** stance, that it's **advisory not authorship proof**, and the borrowed-library credits (LexicalRichness, TextDescriptives, papuGaPT2) + their licenses.
- [ ] **Step 2:** `personal_style_pl/AGENTS.md` — add a line: rich metrics are interpretable/length-normalized; perplexity/TextDescriptives are gated; never emit confident AI/human labels for Polish.
- [ ] **Step 3: Acceptance run:**
```bash
# rich (default) profile + overlay
.venv/bin/python -m personal_style_pl.cli build-profile --samples-dir examples/my_style_samples --output artifacts/profile.joblib --no-stylometrix
.venv/bin/python -m personal_style_pl.cli ai-markers --text-file examples/candidates/draft_b.txt --profile artifacts/profile.joblib --json
# perplexity path (profile carries perplexity baseline; overlay scores perplexity markers)
.venv/bin/python -m personal_style_pl.cli build-profile --samples-dir examples/my_style_samples --output artifacts/profile_ppl.joblib --no-stylometrix --with-perplexity
.venv/bin/python -m personal_style_pl.cli ai-markers --text-file examples/candidates/draft_b.txt --profile artifacts/profile_ppl.joblib --with-perplexity --json
# detector enrichment + full suite
.venv/bin/python heuristic_detector.py --text-file examples/candidates/draft_b.txt --json --rich
.venv/bin/python -m pytest -q
```
Expected: `ai-markers` returns markers (incl. perplexity markers in the `--with-perplexity` run) +
score + PL abstain; `--rich` adds the block; full suite green.
- [ ] **Step 4: Commit + finish branch** (use superpowers:finishing-a-development-branch).
```bash
git add README.md personal_style_pl/AGENTS.md
git commit -m "docs: rich-metrics v2 usage, abstain-on-PL stance, library credits

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Coverage map (spec → task)
- Deps/extras/setup/CI → Task 0
- LexicalRichness + burstiness + n-gram + length-normalized densities → Task 1
- **Wire rich (default) + perplexity (opt-in) into the profile baseline + scorer → Task 1b**
- TextDescriptives (dependency distance / POS / descriptive — NO entropy/perplexity) → Task 2
- AI-direction overlay + OOD/abstain → Task 3
- `ai-markers` CLI + `suggest-edits` integration → Task 4
- papuGaPT2 **robust** perplexity + curvature (gated) → Task 5
- `ai-detector-heuristic --rich` → Task 6
- Interpretable hybrid (`--features rich`) → Task 7
- README/AGENTS/acceptance/finish → Task 8

## Notes / risks (post-audit 2026-06-14)
- **Profile baseline (resolved):** the overlay needs marker features in the profile → Task 1b wires
  rich (default) + perplexity (opt-in) into `build_profile` + `similarity` (gated by `profile.config`).
- **TextDescriptives entropy/perplexity are NaN for Polish** (no lexeme-prob table) → EXCLUDED; the
  `_COLUMNS` set is the verified non-NaN subset; all real perplexity comes from Task 5 (papuGaPT2).
- **Perplexity is ADVISORY, not a separator** on this domain (median PPL human≈50 vs AI-assisted≈54);
  aggregation is robust (median + IQR) to survive PDF-extraction outliers.
- Readability indices English-calibrated → excluded from the primary set.
- papuGaPT2 (~500 MB) fetched by the setup script; the feature degrades gracefully if absent
  (gated extractor + model-gated tests, which skip in light CI).
- Lean detector path stays dependency-free: `--rich` lazy-imports and never breaks the base CLI.
- Light CI installs only `lexicalrichness`; TextDescriptives + perplexity tests are model-gated.
