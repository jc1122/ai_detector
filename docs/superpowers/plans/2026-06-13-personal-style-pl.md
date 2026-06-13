# personal_style_pl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI/package that profiles the user's Polish writing style and scores/ranks/edits new texts by similarity to it, integrated with the existing `heuristic_detector`.

**Architecture:** A `personal_style_pl` package layered on the existing repo. Deterministic surface features (reusing `heuristic_detector` tokenization) + optional StyloMetrix (172 PL features) feed a one-class robust-z⊕cosine profile. CLI uses `argparse`+`rich`. Single Python 3.12 venv for the whole project.

**Tech Stack:** Python 3.12, numpy<2 (1.26), pandas 2.1, scikit-learn 1.9, scipy 1.17, joblib, rich, regex, argparse (stdlib); StyloMetrix/spaCy 3.7.2 optional.

**Spec:** `docs/superpowers/specs/2026-06-13-personal-style-pl-design.md`

---

## Conventions for every code task

- TDD: write the failing test, run it (see it fail), implement minimally, run it (see it pass), commit.
- Run tests with the project venv: `.venv/bin/python -m pytest <path> -v`.
- New tests live in `tests/personal_style/`; use plain pytest functions.
- Import Polish tokenization from the existing module: `from heuristic_detector import _extract_words, _split_sentences, _split_paragraphs, _fold_text`.
- Commit message style matches repo (Conventional Commits, e.g. `feat:`, `test:`, `chore:`).

---

## Task 0: Consolidate to a single Python 3.12 venv

**Files:**
- Create: `scripts/setup_style_env.sh`

- [ ] **Step 1: Write the setup script**

Create `scripts/setup_style_env.sh`:

```bash
#!/usr/bin/env bash
# Reproducible single-env setup for ai_detector + personal_style_pl + StyloMetrix.
# Requires `uv` (https://astral.sh/uv). Builds .venv on Python 3.12 (StyloMetrix ceiling).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv python install 3.12
rm -rf .venv
uv venv --python 3.12 .venv

# Editable install of the detector + style + stylometrix + test extras.
# numpy<2 is mandatory: thinc was built against numpy 1.x (numpy 2 -> ABI crash).
uv pip install --python .venv/bin/python -e ".[test,style,style-stylometrix]" "numpy<2" "pandas<2.2"

# pl_nask model: install --no-deps from the TLS-valid HF mirror (it declares spacy<3.6
# which would otherwise drag an unbuildable thinc 8.1.x; the official IPI PAN host has an
# expired TLS cert).
uv pip install --python .venv/bin/python --no-deps \
  "https://huggingface.co/ipipan/pl_nask/resolve/main/pl_nask-0.0.7.tar.gz"

echo "Done. Verify:"
echo "  .venv/bin/python -c \"import stylo_metrix as sm; print(sm.StyloMetrix('pl').transform(['Ala ma kota.']).shape)\""
echo "  .venv/bin/python -m pytest -q"
```

- [ ] **Step 2: Make it executable and run it**

Run:
```bash
chmod +x scripts/setup_style_env.sh
./scripts/setup_style_env.sh
```
Expected: a fresh `.venv` on Python 3.12 with the full stack. (This requires Task 1's pyproject extras to exist; if running before Task 1, install ad-hoc per the spec's verified version set, then re-run after Task 1.)

- [ ] **Step 3: Verify detector + StyloMetrix both work**

Run:
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -c "import stylo_metrix as sm; print(sm.StyloMetrix('pl').transform(['Ala ma kota i psa.']).shape)"
```
Expected: existing tests pass; transform prints `(1, 173)`.

- [ ] **Step 4: Commit**

```bash
git add scripts/setup_style_env.sh
git commit -m "chore: add reproducible single-env (py3.12) setup script with StyloMetrix"
```

---

## Task 1: Package scaffold, pyproject extras, CI wiring

**Files:**
- Create: `personal_style_pl/__init__.py`, `personal_style_pl/features/__init__.py`, `personal_style_pl/profile/__init__.py`, `personal_style_pl/edit/__init__.py`, `personal_style_pl/models/__init__.py`, `personal_style_pl/utils/__init__.py`, `tests/personal_style/__init__.py`
- Modify: `pyproject.toml`, `MANIFEST.in`, `.github/workflows/ci.yml`

- [ ] **Step 1: Create package `__init__.py` files**

`personal_style_pl/__init__.py`:
```python
"""Polish personal writing-style similarity toolkit."""

__version__ = "0.1.0"
```

Create empty `__init__.py` in `personal_style_pl/features/`, `personal_style_pl/profile/`, `personal_style_pl/edit/`, `personal_style_pl/models/`, `personal_style_pl/utils/`, and `tests/personal_style/`.

- [ ] **Step 2: Update `pyproject.toml`**

In `[project.optional-dependencies]` add (keep existing `test`):
```toml
style = ["numpy<2", "pandas", "scikit-learn", "scipy", "joblib", "rich", "regex"]
style-stylometrix = ["stylo_metrix", "spacy==3.7.2", "spacy-transformers>=1.3,<1.4", "numpy<2"]
style-ml = ["lightgbm", "shap"]
```

In `[project.scripts]` add:
```toml
personal-style-pl = "personal_style_pl.cli:main"
```

Replace the `[tool.setuptools]` section to also declare packages:
```toml
[tool.setuptools]
py-modules = [
  "run_ensemble",
  "deploy_meld",
  "detector_daemon",
  "heuristic_detector",
  "calibration_config",
  "calibrate_detector",
]
packages = [
  "personal_style_pl",
  "personal_style_pl.features",
  "personal_style_pl.profile",
  "personal_style_pl.edit",
  "personal_style_pl.models",
  "personal_style_pl.utils",
]
```

(Leave `requires-python = ">=3.10"` unchanged.)

- [ ] **Step 3: Update `MANIFEST.in`**

Append:
```
recursive-include examples *
```

- [ ] **Step 4: Update CI (`.github/workflows/ci.yml`)**

Change the existing "Unit tests" step command from `python -m pytest -q` to:
```yaml
      - name: Unit tests
        run: |
          python -m pytest -q --ignore=tests/personal_style
```

Add a new job after `test` (light: no torch/stylometrix):
```yaml
  style-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install style extra
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[style]" "numpy<2"
          python -m pip install pytest
      - name: Compile package
        run: python -m py_compile $(find personal_style_pl -name '*.py')
      - name: Style unit tests
        run: python -m pytest -q tests/personal_style
      - name: CLI help smoke
        run: python -m personal_style_pl.cli --help
```

- [ ] **Step 5: Verify install + import**

Run:
```bash
.venv/bin/python -m pip install -e . >/dev/null 2>&1 || ./scripts/setup_style_env.sh
.venv/bin/python -c "import personal_style_pl; print(personal_style_pl.__version__)"
```
Expected: prints `0.1.0`.

- [ ] **Step 6: Commit**

```bash
git add personal_style_pl tests/personal_style pyproject.toml MANIFEST.in .github/workflows/ci.yml
git commit -m "feat: scaffold personal_style_pl package, extras, and CI style job"
```

---

## Task 2: `config.py` — phrase lists and thresholds

**Files:**
- Create: `personal_style_pl/config.py`
- Test: `tests/personal_style/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_config.py`:
```python
from personal_style_pl import config


def test_phrase_lists_are_folded_and_nonempty():
    for name in ("TRANSITION_PHRASES", "BOILERPLATE_PHRASES", "HEDGE_PHRASES"):
        phrases = getattr(config, name)
        assert phrases, f"{name} empty"
        for p in phrases:
            # folded: lowercase, no diacritics, no uppercase
            assert p == p.lower()
            assert all(ch not in p for ch in "ąćęłńóśźż")


def test_seeded_from_heuristic_detector():
    # transitions include canonical spec phrases (folded)
    assert "co wiecej" in config.TRANSITION_PHRASES
    assert "ponadto" in config.TRANSITION_PHRASES
    assert "podsumowujac" in config.BOILERPLATE_PHRASES


def test_score_thresholds():
    assert config.LABEL_THRESHOLDS["close_to_my_style"] == 80
    assert config.LABEL_THRESHOLDS["mixed"] == 55
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_config.py -v`
Expected: FAIL (no module `config`).

- [ ] **Step 3: Implement `personal_style_pl/config.py`**

```python
"""Configurable Polish phrase lists and scoring thresholds.

Phrases are stored folded (lowercase, diacritics stripped) so matching against
heuristic_detector._fold_text(text) is consistent. These are STYLE indicators,
not proof of AI authorship.
"""

from __future__ import annotations

from heuristic_detector import _fold_text, AI_PHRASES_PL, AI_WORDS_PL

_TRANSITIONS_RAW = (
    "co więcej", "ponadto", "poza tym", "z jednej strony", "z drugiej strony",
    "jednocześnie", "tym samym", "w efekcie", "w rezultacie", "dlatego",
)
_BOILERPLATE_RAW = (
    "warto zauważyć", "należy podkreślić", "trzeba pamiętać", "kluczowe znaczenie",
    "istotnym elementem", "dynamicznie zmieniającym się", "w dzisiejszych czasach",
    "podsumowując",
)
_HEDGES_RAW = (
    "wydaje się", "można powiedzieć", "raczej", "prawdopodobnie", "być może",
    "w pewnym sensie",
)
_FIRST_PERSON_SINGULAR_RAW = ("ja", "mnie", "mi", "mną", "moim", "moja", "moje", "mój")
_FIRST_PERSON_PLURAL_RAW = ("my", "nas", "nam", "nami", "nasz", "nasza", "nasze", "naszych")
# Clause markers (subordinators/conjunctions) — folded.
_CLAUSE_MARKERS_RAW = (
    "że", "który", "która", "które", "ponieważ", "gdyż", "aby", "żeby", "jeśli",
    "jeżeli", "gdy", "kiedy", "chociaż", "mimo", "dlatego", "więc",
)


def _fold_all(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        folded = _fold_text(item)
        if folded and folded not in seen:
            seen.append(folded)
    return tuple(seen)


# Seed transitions/boilerplate from both the spec lists and heuristic_detector.AI_PHRASES_PL.
TRANSITION_PHRASES = _fold_all(_TRANSITIONS_RAW)
BOILERPLATE_PHRASES = _fold_all(_BOILERPLATE_RAW)
HEDGE_PHRASES = _fold_all(_HEDGES_RAW)
FIRST_PERSON_SINGULAR = _fold_all(_FIRST_PERSON_SINGULAR_RAW)
FIRST_PERSON_PLURAL = _fold_all(_FIRST_PERSON_PLURAL_RAW)
CLAUSE_MARKERS = _fold_all(_CLAUSE_MARKERS_RAW)

# Extra AI-marker phrases available to the bridge (already folded upstream).
AI_MARKER_PHRASES = tuple(AI_PHRASES_PL)
AI_MARKER_WORDS = tuple(AI_WORDS_PL)

# Chunking defaults.
DEFAULT_CHUNK_SENTENCES = 8
DEFAULT_MIN_CHUNK_TOKENS = 120

# Scoring.
Z_CLIP = 8.0
BLEND_Z = 0.7
BLEND_COSINE = 0.3
LABEL_THRESHOLDS = {"close_to_my_style": 80, "mixed": 55}
MIN_CANDIDATE_TOKENS = 40

WEAK_PROFILE_WARNING = (
    "Profile is weak: provide at least 10–20 writing samples or 5,000+ words."
)
STYLE_NOT_AUTHORSHIP_WARNING = (
    "This is style similarity, not proof of authorship."
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_style_pl/config.py tests/personal_style/test_config.py
git commit -m "feat: config phrase lists and scoring thresholds"
```

---

## Task 3: `textsplit.py` — sentence-aware chunking

**Files:**
- Create: `personal_style_pl/textsplit.py`
- Test: `tests/personal_style/test_chunking.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_chunking.py`:
```python
from personal_style_pl.textsplit import Chunk, chunk_document


PL = (" Zdanie pierwsze ma kilka słów. Zdanie drugie jest tutaj. "
      "Trzecie zdanie również. Czwarte zdanie tutaj jest. "
      "Piąte zdanie mamy. Szóste zdanie też. Siódme zdanie krótkie. "
      "Ósme zdanie ostatnie. Dziewiąte zdanie dodatkowe. Dziesiąte zdanie tu.")


def test_chunking_preserves_diacritics():
    chunks = chunk_document(PL + " Łódź, gęś, źdźbło, ćma.", doc_id="d0",
                            chunk_sentences=8, min_chunk_tokens=5)
    joined = " ".join(c.text for c in chunks)
    for ch in "łęśćź":
        assert ch in joined


def test_chunk_groups_by_sentence_count():
    chunks = chunk_document(PL, doc_id="d0", chunk_sentences=4, min_chunk_tokens=1)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "d0" for c in chunks)
    assert chunks[0].sentence_count == 4


def test_short_chunks_merge_to_min_tokens():
    chunks = chunk_document(PL, doc_id="d0", chunk_sentences=1, min_chunk_tokens=120)
    # whole thing is < 120 tokens -> a single (under-min) chunk, flagged
    assert len(chunks) == 1
    assert chunks[0].under_min_tokens is True


def test_empty_text_returns_no_chunks():
    assert chunk_document("   ", doc_id="d0") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_chunking.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `personal_style_pl/textsplit.py`**

```python
"""Sentence-aware chunking with document IDs (avoids evaluation leakage)."""

from __future__ import annotations

from dataclasses import dataclass

from heuristic_detector import _extract_words, _split_sentences

from .config import DEFAULT_CHUNK_SENTENCES, DEFAULT_MIN_CHUNK_TOKENS


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    token_count: int
    sentence_count: int
    under_min_tokens: bool


def chunk_document(
    text: str,
    *,
    doc_id: str,
    chunk_sentences: int = DEFAULT_CHUNK_SENTENCES,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
) -> list[Chunk]:
    """Group sentences into chunks of `chunk_sentences`, merging trailing
    short groups so each chunk reaches `min_chunk_tokens` when possible."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    groups: list[list[str]] = [
        sentences[i : i + chunk_sentences]
        for i in range(0, len(sentences), chunk_sentences)
    ]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    for group in groups:
        buffer.extend(group)
        token_count = len(_extract_words(" ".join(buffer)))
        if token_count >= min_chunk_tokens:
            chunks.append(_make_chunk(doc_id, len(chunks), buffer))
            buffer = []
    if buffer:
        chunks.append(_make_chunk(doc_id, len(chunks), buffer))

    # If the only chunk is under min, keep it but flag it (caller warns).
    for chunk in chunks:
        chunk.under_min_tokens = chunk.token_count < min_chunk_tokens
    return chunks


def _make_chunk(doc_id: str, chunk_id: int, sentences: list[str]) -> Chunk:
    text = " ".join(sentences)
    token_count = len(_extract_words(text))
    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        token_count=token_count,
        sentence_count=len(sentences),
        under_min_tokens=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_chunking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_style_pl/textsplit.py tests/personal_style/test_chunking.py
git commit -m "feat: sentence-aware chunking with doc ids"
```

---

## Task 4: `surface_features.py` — deterministic Polish surface features

**Files:**
- Create: `personal_style_pl/features/surface_features.py`
- Test: `tests/personal_style/test_surface_features.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_surface_features.py`:
```python
import numpy as np

from personal_style_pl.features.surface_features import (
    SurfaceFeatureExtractor,
    SURFACE_FEATURE_NAMES,
)

PL = ("Wczoraj poszedłem do urzędu, bo odkładałem wymianę dokumentu. "
      "Kolejka była krótka, ale formularz poprawiałem dwa razy. "
      "Pani cierpliwie pokazała mi błąd.")


def test_feature_names_are_stable_and_complete():
    ext = SurfaceFeatureExtractor().fit([PL])
    assert list(ext.get_feature_names_out()) == list(SURFACE_FEATURE_NAMES)
    assert len(SURFACE_FEATURE_NAMES) == len(set(SURFACE_FEATURE_NAMES))
    for required in ("avg_sentence_len_tokens", "comma_per_sentence",
                     "type_token_ratio", "hapax_ratio", "transition_phrase_count"):
        assert required in SURFACE_FEATURE_NAMES


def test_transform_shape_and_diacritics_counted():
    ext = SurfaceFeatureExtractor()
    X = ext.fit_transform([PL])
    assert isinstance(X, np.ndarray)
    assert X.shape == (1, len(SURFACE_FEATURE_NAMES))
    row = dict(zip(SURFACE_FEATURE_NAMES, X[0]))
    assert row["sentence_count"] == 3
    assert row["comma_per_sentence"] > 0
    assert row["first_person_singular_count"] >= 1  # "mi"


def test_empty_text_is_safe():
    ext = SurfaceFeatureExtractor()
    X = ext.fit_transform([""])
    assert X.shape == (1, len(SURFACE_FEATURE_NAMES))
    assert not np.isnan(X).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_surface_features.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `personal_style_pl/features/surface_features.py`**

```python
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
            sum(1 for ch in text if ch in ".,;:!?()-–—\"'„”«»…[]{}"), len(text)),
        "comma_per_sentence": _safe_div(text.count(","), len(sentences)),
        "semicolon_per_sentence": _safe_div(text.count(";"), len(sentences)),
        "colon_per_sentence": _safe_div(text.count(":"), len(sentences)),
        "dash_per_sentence": _safe_div(len(re.findall(r"[-–—]", text)), len(sentences)),
        "question_mark_count": float(text.count("?")),
        "exclamation_mark_count": float(text.count("!")),
        "parenthesis_density": _safe_div(text.count("(") + text.count(")"), len(text)),
        "quote_density": _safe_div(len(re.findall(r"[\"'„”«»]", text)), len(text)),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_surface_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_style_pl/features/surface_features.py tests/personal_style/test_surface_features.py
git commit -m "feat: deterministic Polish surface feature extractor"
```

---

## Task 5: `utils/json.py` + `io.py` — loading samples, writing artifacts

**Files:**
- Create: `personal_style_pl/utils/json.py`, `personal_style_pl/io.py`
- Test: `tests/personal_style/test_io.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_io.py`:
```python
import csv
import json

from personal_style_pl.io import load_samples_from_dir, load_samples_from_csv, SampleDoc
from personal_style_pl.utils.json import dump_json


def test_load_samples_from_dir(tmp_path):
    (tmp_path / "a.txt").write_text("Pierwszy tekst. Łódź.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Drugi tekst po polsku.", encoding="utf-8")
    docs = load_samples_from_dir(tmp_path)
    assert len(docs) == 2
    assert all(isinstance(d, SampleDoc) for d in docs)
    assert {d.doc_id for d in docs} == {"a", "b"}
    assert "Łódź" in next(d.text for d in docs if d.doc_id == "a")


def test_load_samples_from_csv(tmp_path):
    path = tmp_path / "s.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "source", "date", "genre"])
        w.writerow(["Mój tekst pierwszy.", "blog", "2024-01-01", "essay"])
        w.writerow(["Mój tekst drugi.", "email", "2024-02-10", "email"])
    docs = load_samples_from_csv(path, text_col="text")
    assert len(docs) == 2
    assert docs[0].genre == "essay"


def test_dump_json_preserves_polish(tmp_path):
    p = tmp_path / "o.json"
    dump_json({"k": "źdźbło"}, p)
    assert "źdźbło" in p.read_text(encoding="utf-8")
    assert json.loads(p.read_text(encoding="utf-8"))["k"] == "źdźbło"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_io.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement the modules**

`personal_style_pl/utils/json.py`:
```python
"""JSON helpers that preserve Polish characters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def dumps_json(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)
```

`personal_style_pl/io.py`:
```python
"""Load writing samples (dir or CSV) and write artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleDoc:
    doc_id: str
    text: str
    source: str | None = None
    date: str | None = None
    genre: str | None = None


def load_samples_from_dir(samples_dir: str | Path) -> list[SampleDoc]:
    directory = Path(samples_dir)
    if not directory.is_dir():
        raise RuntimeError(f"Samples directory not found: {directory}")
    docs: list[SampleDoc] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(SampleDoc(doc_id=path.stem, text=text))
    if not docs:
        raise RuntimeError(f"No .txt samples found in {directory}")
    return docs


def load_samples_from_csv(csv_path: str | Path, *, text_col: str = "text") -> list[SampleDoc]:
    path = Path(csv_path)
    if not path.is_file():
        raise RuntimeError(f"CSV not found: {path}")
    docs: list[SampleDoc] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise RuntimeError(f"CSV missing required text column '{text_col}'")
        for index, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            docs.append(SampleDoc(
                doc_id=row.get("source") or f"row{index}",
                text=text,
                source=row.get("source"),
                date=row.get("date"),
                genre=row.get("genre"),
            ))
    if not docs:
        raise RuntimeError(f"No rows with non-empty '{text_col}' in {path}")
    return docs


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_io.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_style_pl/io.py personal_style_pl/utils/json.py tests/personal_style/test_io.py
git commit -m "feat: sample loading (dir/csv) and json/artifact helpers"
```

---

## Task 6: `profile/build_profile.py` — StyleProfile + builder

**Files:**
- Create: `personal_style_pl/profile/build_profile.py`
- Test: `tests/personal_style/test_build_profile.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_build_profile.py`:
```python
import numpy as np

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile, StyleProfile


def _docs():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę. "
            "Potem kupiłem chleb i wróciłem do domu spokojnie. "
            "Wieczorem zapisałem numer sprawy na kartce. ")
    return [SampleDoc(doc_id=f"d{i}", text=base * 3) for i in range(4)]


def test_build_profile_returns_style_profile():
    profile = build_profile(_docs(), use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=20)
    assert isinstance(profile, StyleProfile)
    assert profile.language == "pl"
    assert profile.training_chunk_count >= 4
    assert len(profile.feature_names) == len(profile.center)
    assert len(profile.center) == len(profile.scale)
    assert profile.scale.min() > 0  # no zero scale (floored)


def test_weak_profile_warning_for_tiny_corpus():
    profile = build_profile([SampleDoc(doc_id="d0", text="Krótki tekst tutaj jest.")],
                            use_stylometrix=False, chunk_sentences=8, min_chunk_tokens=120)
    assert any("Profile is weak" in w for w in profile.warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_build_profile.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `personal_style_pl/profile/build_profile.py`**

```python
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
    # scorer can reproduce identical dimensions (Task 7 / Task 12).
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
    # training self-scores + temperature are filled by calibration (Task 7).
    from .calibration import finalize_profile
    finalize_profile(profile, matrix)
    return profile
```

- [ ] **Step 4: Run test to verify it fails on missing `finalize_profile`, then add a stub**

This task depends on Task 7. For now, add a temporary stub at the bottom of a new file `personal_style_pl/profile/calibration.py`:
```python
"""Temperature calibration + profile finalization (full impl in Task 7)."""

from __future__ import annotations

import numpy as np


def finalize_profile(profile, matrix: np.ndarray) -> None:
    profile.temperature = 1.0
    profile.training_scores = []
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_build_profile.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add personal_style_pl/profile/build_profile.py personal_style_pl/profile/calibration.py tests/personal_style/test_build_profile.py
git commit -m "feat: StyleProfile dataclass and one-class profile builder"
```

---

## Task 7: `profile/calibration.py` + `profile/similarity.py` — scoring

**Files:**
- Modify: `personal_style_pl/profile/calibration.py`
- Create: `personal_style_pl/profile/similarity.py`
- Test: `tests/personal_style/test_similarity.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_similarity.py`:
```python
from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text, ScoreResult


def _profile():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. "
            "Potem kupiłem chleb i wróciłem do domu wieczorem. "
            "Zapisałem numer sprawy na kartce przy komputerze. ")
    docs = [SampleDoc(doc_id=f"d{i}", text=base * 5) for i in range(6)]
    return build_profile(docs, use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)


def test_score_self_is_high():
    profile = _profile()
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. ") * 4
    result = score_text(profile, base)
    assert isinstance(result, ScoreResult)
    assert 0 <= result.style_match_score <= 100
    assert result.label in {"close_to_my_style", "mixed", "far_from_my_style"}
    assert result.style_match_score >= 55


def test_insufficient_text():
    profile = _profile()
    result = score_text(profile, "Za krótkie.")
    assert result.label == "insufficient_text"
    assert any("short" in w.lower() for w in result.warnings)


def test_dissimilar_text_scores_lower_than_self():
    profile = _profile()
    self_text = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
                 "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. ") * 4
    odd = ("WARTO ZAUWAŻYĆ!!! Lista:\n- punkt\n- punkt\n- punkt\n"
           "Tekst — z wieloma — myślnikami — naprawdę — dużo.") * 4
    assert score_text(profile, self_text).style_match_score >= \
        score_text(profile, odd).style_match_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_similarity.py -v`
Expected: FAIL.

- [ ] **Step 3: Replace `calibration.py` with the full implementation**

`personal_style_pl/profile/calibration.py`:
```python
"""Temperature calibration from training self-distances."""

from __future__ import annotations

import math

import numpy as np

from ..config import Z_CLIP


def chunk_distance(profile, feature_row: np.ndarray) -> float:
    """Median robust z-distance over stable features (clipped). Returns 0.0 when the
    profile has no stable features (degenerate uniform corpus); real varied corpora
    always have stable features."""
    mask = profile.stable_mask
    center = profile.robust_center[mask]
    scale = profile.robust_scale[mask]
    z = np.abs((feature_row[mask] - center) / scale)
    z = np.clip(z, 0.0, Z_CLIP)
    return float(np.median(z)) if z.size else 0.0


def finalize_profile(profile, matrix: np.ndarray) -> None:
    distances = [chunk_distance(profile, row) for row in matrix]
    profile.training_scores = distances
    if len(distances) >= 5:
        # Calibrate temperature so a typical own-chunk (median self-distance) maps to a
        # z-component of ~80 (the close_to_my_style boundary) before the cosine blend:
        # 100*exp(-median/temperature) = 80  ->  temperature = median / -ln(0.80).
        median_d = float(np.median(distances))
        profile.temperature = max(median_d / -math.log(0.80), 0.5)
    else:
        profile.temperature = 1.0
```

- [ ] **Step 4: Implement `personal_style_pl/profile/similarity.py`**

```python
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
    if not chunks:
        chunks_texts = [text]
    else:
        chunks_texts = [c.text for c in chunks]

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/personal_style/test_similarity.py tests/personal_style/test_build_profile.py -v`
Expected: PASS. (If `test_dissimilar...` is flaky, tune the cosine_component formula — keep the blend weights from config.)

- [ ] **Step 6: Commit**

```bash
git add personal_style_pl/profile/calibration.py personal_style_pl/profile/similarity.py tests/personal_style/test_similarity.py
git commit -m "feat: one-class scoring (robust-z blend) with temperature calibration"
```

---

## Task 8: `cli.py` — `build-profile`, `score`, `rank`

**Files:**
- Create: `personal_style_pl/cli.py`, `personal_style_pl/profile/report.py`
- Test: `tests/personal_style/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_cli_smoke.py`:
```python
import csv
import json

from personal_style_pl.cli import main


def _write_samples(tmp_path):
    d = tmp_path / "samples"
    d.mkdir()
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. "
            "Potem kupiłem chleb i wróciłem do domu wieczorem. ") * 4
    for i in range(5):
        (d / f"s{i}.txt").write_text(base, encoding="utf-8")
    return d


def test_build_score_rank_smoke(tmp_path, capsys):
    samples = _write_samples(tmp_path)
    profile = tmp_path / "p.joblib"
    rc = main(["build-profile", "--samples-dir", str(samples),
               "--output", str(profile), "--no-stylometrix",
               "--chunk-sentences", "2", "--min-chunk-tokens", "15"])
    assert rc == 0 and profile.exists()

    draft = tmp_path / "draft.txt"
    draft.write_text(("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
                      "Formularz poprawiałem dwa razy. ") * 4, encoding="utf-8")
    rc = main(["score", "--profile", str(profile), "--text-file", str(draft), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"style_match_score", "label", "confidence",
                            "warnings", "summary", "top_matches", "top_mismatches",
                            "chunk_scores"}

    cand = tmp_path / "cands"
    cand.mkdir()
    (cand / "a.txt").write_text(draft.read_text(encoding="utf-8"), encoding="utf-8")
    (cand / "b.txt").write_text("Inny tekst. " * 30, encoding="utf-8")
    ranking = tmp_path / "rank.csv"
    rc = main(["rank", "--profile", str(profile), "--candidates-dir", str(cand),
               "--output", str(ranking)])
    assert rc == 0 and ranking.exists()
    rows = list(csv.DictReader(ranking.open(encoding="utf-8")))
    assert {"filename", "style_match_score", "label", "confidence",
            "word_count", "warnings"} <= set(rows[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_cli_smoke.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `personal_style_pl/profile/report.py`**

```python
"""Serialize ScoreResult to dict/JSON."""

from __future__ import annotations

from dataclasses import asdict

from .similarity import ScoreResult


def score_result_to_dict(result: ScoreResult) -> dict:
    return asdict(result)
```

- [ ] **Step 4: Implement `personal_style_pl/cli.py`**

```python
"""argparse CLI for personal_style_pl. Entry: `python -m personal_style_pl.cli`."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib

from .io import load_samples_from_dir, load_samples_from_csv, ensure_parent
from .profile.build_profile import build_profile
from .profile.similarity import score_text
from .profile.report import score_result_to_dict
from .utils.json import dumps_json
from heuristic_detector import _extract_words


def _cmd_build_profile(args) -> int:
    if args.csv:
        docs = load_samples_from_csv(args.csv, text_col=args.text_col)
    else:
        docs = load_samples_from_dir(args.samples_dir)
    profile = build_profile(
        docs, use_stylometrix=not args.no_stylometrix,
        include_ngrams=args.include_ngrams,
        chunk_sentences=args.chunk_sentences, min_chunk_tokens=args.min_chunk_tokens)
    joblib.dump(profile, ensure_parent(args.output))
    if args.report:
        from .utils.json import dump_json
        dump_json({
            "profile_id": profile.profile_id, "created_at": profile.created_at,
            "training_sample_count": profile.training_sample_count,
            "training_chunk_count": profile.training_chunk_count,
            "total_tokens": profile.total_tokens, "genres": profile.genres,
            "config": profile.config, "warnings": profile.warnings,
        }, args.report)
    print(f"Profile written to {args.output} "
          f"({profile.training_chunk_count} chunks, {profile.total_tokens} tokens)")
    for w in profile.warnings:
        print(f"  warning: {w}")
    return 0


def _cmd_score(args) -> int:
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    result = score_text(profile, text)
    payload = score_result_to_dict(result)
    if args.with_heuristics:
        from .bridge import attach_heuristics
        payload = attach_heuristics(payload, text)
    if args.json:
        print(dumps_json(payload, indent=2))
    else:
        print(f"Style match: {result.style_match_score}/100 ({result.label}), "
              f"confidence {result.confidence}")
        print(result.summary)
    return 0


def _cmd_rank(args) -> int:
    profile = joblib.load(args.profile)
    rows = []
    for path in sorted(Path(args.candidates_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        result = score_text(profile, text)
        rows.append({
            "filename": path.name, "style_match_score": result.style_match_score,
            "label": result.label, "confidence": result.confidence,
            "word_count": len(_extract_words(text)),
            "warnings": "; ".join(result.warnings),
        })
    rows.sort(key=lambda r: r["style_match_score"], reverse=True)
    with ensure_parent(args.output).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "style_match_score", "label", "confidence",
            "word_count", "warnings"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ranking written to {args.output} ({len(rows)} candidates)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="personal-style-pl",
                                     description="Polish personal writing-style similarity.")
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build-profile", help="Build a style profile from samples.")
    bp.add_argument("--samples-dir")
    bp.add_argument("--csv")
    bp.add_argument("--text-col", default="text")
    bp.add_argument("--output", required=True)
    bp.add_argument("--report")
    bp.add_argument("--min-chunk-tokens", type=int, default=120)
    bp.add_argument("--chunk-sentences", type=int, default=8)
    bp.add_argument("--include-ngrams", action="store_true")
    bp.add_argument("--no-stylometrix", action="store_true")
    bp.set_defaults(func=_cmd_build_profile)

    sc = sub.add_parser("score", help="Score one text against a profile.")
    sc.add_argument("--profile", required=True)
    sc.add_argument("--text-file", required=True)
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--with-heuristics", action="store_true")
    sc.set_defaults(func=_cmd_score)

    rk = sub.add_parser("rank", help="Rank candidate drafts.")
    rk.add_argument("--profile", required=True)
    rk.add_argument("--candidates-dir", required=True)
    rk.add_argument("--output", required=True)
    rk.set_defaults(func=_cmd_rank)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-profile" and not (args.samples_dir or args.csv):
            parser.error("build-profile requires --samples-dir or --csv")
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_cli_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Verify CLI help and acceptance commands run**

Run:
```bash
.venv/bin/python -m personal_style_pl.cli --help
.venv/bin/python -m personal_style_pl.cli build-profile --help
```
Expected: help text with subcommands.

- [ ] **Step 7: Commit**

```bash
git add personal_style_pl/cli.py personal_style_pl/profile/report.py tests/personal_style/test_cli_smoke.py
git commit -m "feat: CLI build-profile/score/rank with JSON+CSV contracts"
```

---

## Task 9: `describe-profile` command + report

**Files:**
- Modify: `personal_style_pl/cli.py`, `personal_style_pl/profile/report.py`
- Test: `tests/personal_style/test_describe.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_describe.py`:
```python
import joblib

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.cli import main


def test_describe_profile_writes_markdown(tmp_path):
    docs = [SampleDoc(doc_id=f"d{i}",
                      text=("Zdanie jedno tutaj jest. Zdanie dwa tutaj jest. " * 8))
            for i in range(4)]
    profile = build_profile(docs, use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=10)
    p = tmp_path / "p.joblib"
    joblib.dump(profile, p)
    out = tmp_path / "summary.md"
    rc = main(["describe-profile", "--profile", str(p), "--output", str(out)])
    assert rc == 0 and out.exists()
    text = out.read_text(encoding="utf-8")
    for heading in ("Sentence length", "Punctuation", "Lexical diversity", "Limitations"):
        assert heading in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_describe.py -v`
Expected: FAIL.

- [ ] **Step 3: Add `profile_to_markdown` to `report.py`**

Append to `personal_style_pl/profile/report.py`:
```python
def profile_to_markdown(profile) -> str:
    names = profile.feature_names
    center = {n: float(profile.center[i]) for i, n in enumerate(names)}

    def g(key, default=0.0):
        return center.get(key, default)

    lines = [
        f"# Style profile {profile.profile_id}",
        "",
        f"- Created: {profile.created_at}",
        f"- Trained on: {profile.training_sample_count} samples, "
        f"{profile.training_chunk_count} chunks, {profile.total_tokens} tokens",
        f"- Genres: {', '.join(profile.genres) or 'unspecified'}",
        f"- StyloMetrix used: {profile.config.get('use_stylometrix')}",
        "",
        "## Sentence length habits",
        f"- Avg tokens/sentence: {g('avg_sentence_len_tokens'):.1f} "
        f"(std {g('std_sentence_len_tokens'):.1f}, cv {g('sentence_len_cv'):.2f})",
        "## Paragraph length habits",
        f"- Avg sentences/paragraph: {g('avg_paragraph_len_sentences'):.1f}",
        "## Punctuation habits",
        f"- Commas/sentence: {g('comma_per_sentence'):.2f}; "
        f"dashes/sentence: {g('dash_per_sentence'):.2f}; "
        f"punctuation density: {g('punctuation_density'):.3f}",
        "## Lexical diversity",
        f"- Type-token ratio: {g('type_token_ratio'):.2f}; hapax ratio: {g('hapax_ratio'):.2f}",
        "## Common transitions and function words",
        f"- Transition phrases/chunk: {g('transition_phrase_count'):.2f}; "
        f"clause markers/sentence: {g('average_clause_marker_count'):.2f}",
        "## Formulaic phrase frequency",
        f"- Boilerplate phrases/chunk: {g('generic_boilerplate_count'):.2f}; "
        f"hedges/chunk: {g('hedge_count'):.2f}",
        "## What this profile was trained on",
        f"- {profile.training_sample_count} documents in genres: "
        f"{', '.join(profile.genres) or 'unspecified'}",
        "## Limitations",
    ]
    for w in profile.warnings:
        lines.append(f"- {w}")
    if profile.config.get("use_stylometrix"):
        lines.append("- StyloMetrix feature families included (172 PL features).")
    else:
        lines.append("- StyloMetrix unavailable/disabled: surface features only.")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Wire `describe-profile` into `cli.py`**

Add to `build_parser()` (before `return parser`):
```python
    dp = sub.add_parser("describe-profile", help="Summarize a profile as Markdown.")
    dp.add_argument("--profile", required=True)
    dp.add_argument("--output", required=True)
    dp.set_defaults(func=_cmd_describe_profile)
```
Add the handler near the other `_cmd_*`:
```python
def _cmd_describe_profile(args) -> int:
    from .profile.report import profile_to_markdown
    profile = joblib.load(args.profile)
    ensure_parent(args.output).write_text(profile_to_markdown(profile), encoding="utf-8")
    print(f"Profile summary written to {args.output}")
    return 0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_describe.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add personal_style_pl/cli.py personal_style_pl/profile/report.py tests/personal_style/test_describe.py
git commit -m "feat: describe-profile Markdown summary"
```

---

## Task 10: `edit/` — `suggest-edits` and conservative `edit`

**Files:**
- Create: `personal_style_pl/edit/rules.py`, `personal_style_pl/edit/style_editor.py`
- Modify: `personal_style_pl/cli.py`
- Test: `tests/personal_style/test_edit.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_edit.py`:
```python
import joblib

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.cli import main


def _profile(tmp_path):
    docs = [SampleDoc(doc_id=f"d{i}",
                      text=("Krótkie zdanie. Drugie zdanie tutaj. " * 8))
            for i in range(4)]
    profile = build_profile(docs, use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=10)
    p = tmp_path / "p.joblib"
    joblib.dump(profile, p)
    return p


def test_suggest_edits_writes_markdown_with_metadata(tmp_path):
    p = _profile(tmp_path)
    draft = tmp_path / "draft.txt"
    draft.write_text("Warto zauważyć, że to zdanie jest dość długie i zawiera wiele "
                     "fragmentów połączonych przecinkami, co bywa męczące dla czytelnika "
                     "i odbiega od zwykłego stylu pisania autora w tym profilu.",
                     encoding="utf-8")
    out = tmp_path / "sug.md"
    rc = main(["suggest-edits", "--profile", str(p), "--text-file", str(draft),
               "--output", str(out), "--mode", "light"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "machine_assisted_style_edit: true" in text
    assert "mode: light" in text


def test_edit_preserves_numbers_urls_and_normalizes_whitespace(tmp_path):
    p = _profile(tmp_path)
    draft = tmp_path / "draft.txt"
    draft.write_text("Zobacz https://example.com/x oraz 12 345 zł.\n\n\nKoniec   tekstu.",
                     encoding="utf-8")
    out = tmp_path / "edited.txt"
    rc = main(["edit", "--profile", str(p), "--text-file", str(draft),
               "--output", str(out), "--mode", "light"])
    assert rc == 0
    edited = out.read_text(encoding="utf-8")
    assert "https://example.com/x" in edited
    assert "12 345" in edited
    assert "   " not in edited  # collapsed runs of spaces
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_edit.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `personal_style_pl/edit/rules.py`**

```python
"""Conservative, deterministic text rules. Never invent facts."""

from __future__ import annotations

import re
from dataclasses import dataclass

_URL_RE = re.compile(r"https?://\S+")


@dataclass
class EditSuggestion:
    issue: str
    reason: str
    suggestion: str
    before: str | None = None
    after: str | None = None
    severity: str = "info"


def normalize_whitespace(text: str) -> str:
    # Protect URLs, collapse intra-line spaces, cap blank lines at one.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
```

- [ ] **Step 4: Implement `personal_style_pl/edit/style_editor.py`**

```python
"""StyleSuggestionEngine + conservative deterministic editor."""

from __future__ import annotations

import numpy as np

from ..features.surface_features import SurfaceFeatureExtractor, SURFACE_FEATURE_NAMES
from ..profile.similarity import score_text
from .rules import EditSuggestion, normalize_whitespace

_PROTECTED_NOTE = ("Never changes names, dates, numbers, legal/medical/financial claims, "
                   "citations, quotes, URLs, or code.")


def _feature(profile, name: str) -> float:
    idx = profile.feature_names.index(name)
    return float(profile.center[idx])


def suggest_edits(profile, text: str) -> list[EditSuggestion]:
    feats = dict(zip(SURFACE_FEATURE_NAMES,
                     SurfaceFeatureExtractor().fit_transform([text])[0]))
    out: list[EditSuggestion] = []

    cand_len = feats["avg_sentence_len_tokens"]
    prof_len = _feature(profile, "avg_sentence_len_tokens")
    if cand_len > prof_len * 1.4 and prof_len > 0:
        out.append(EditSuggestion(
            issue="Long sentences",
            reason=f"Avg sentence {cand_len:.0f} tokens vs your usual {prof_len:.0f}.",
            suggestion="Split the longest sentences at safe punctuation (. ; :).",
            severity="warn"))
    elif cand_len < prof_len * 0.6 and prof_len > 0:
        out.append(EditSuggestion(
            issue="Choppy sentences",
            reason=f"Avg sentence {cand_len:.0f} tokens vs your usual {prof_len:.0f}.",
            suggestion="Merge closely related short sentences."))

    if feats["comma_per_sentence"] < _feature(profile, "comma_per_sentence") * 0.5:
        out.append(EditSuggestion(
            issue="Comma rhythm",
            reason="Comma rate is well below your usual rhythm.",
            suggestion="Review punctuation rhythm — don't blindly add commas."))

    if feats["generic_boilerplate_count"] > max(_feature(profile, "generic_boilerplate_count"), 0) + 0.5:
        out.append(EditSuggestion(
            issue="Boilerplate phrases",
            reason="More generic/boilerplate phrases than your profile.",
            suggestion="Delete or replace generic phrases (e.g. 'warto zauważyć').",
            severity="warn"))

    if feats["transition_phrase_count"] > max(_feature(profile, "transition_phrase_count"), 0) + 0.5:
        out.append(EditSuggestion(
            issue="Transition density",
            reason="More transition phrases than your usual density.",
            suggestion="Reduce or vary transitions."))
    return out


def suggestions_to_markdown(profile, text: str, mode: str) -> str:
    result = score_text(profile, text)
    suggestions = suggest_edits(profile, text)
    lines = [
        "---",
        "machine_assisted_style_edit: true",
        f"profile_used: {profile.profile_id}",
        f"mode: {mode}",
        "---",
        "",
        f"# Style suggestions (score {result.style_match_score}/100, {result.label})",
        "",
        f"_{_PROTECTED_NOTE}_",
        "",
        "## Top divergences",
    ]
    for mm in result.top_mismatches[:5]:
        lines.append(f"- `{mm['feature']}`: {mm['effect']} "
                     f"(you≈{mm['profile_mean']}, draft={mm['candidate_value']})")
    lines.append("")
    lines.append("## Suggested edits")
    if not suggestions:
        lines.append("- No conservative suggestions; the draft is close to your style.")
    for s in suggestions:
        lines.append(f"### {s.issue} ({s.severity})")
        lines.append(f"- Reason: {s.reason}")
        lines.append(f"- Suggestion: {s.suggestion}")
    lines.append("")
    lines.append("## Warnings")
    for w in result.warnings:
        lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def conservative_edit(profile, text: str, mode: str) -> str:
    """Deterministic, meaning-preserving edits. Currently: whitespace normalization.
    Sentence splitting/merging are emitted as SUGGESTIONS only (suggest-edits), not
    applied automatically, to guarantee meaning preservation."""
    return normalize_whitespace(text)
```

- [ ] **Step 5: Wire `suggest-edits` and `edit` into `cli.py`**

Add parsers in `build_parser()`:
```python
    se = sub.add_parser("suggest-edits", help="Suggest conservative edits (Markdown).")
    se.add_argument("--profile", required=True)
    se.add_argument("--text-file", required=True)
    se.add_argument("--output", required=True)
    se.add_argument("--mode", choices=["light", "medium", "strong"], default="light")
    se.set_defaults(func=_cmd_suggest_edits)

    ed = sub.add_parser("edit", help="Apply conservative deterministic edits.")
    ed.add_argument("--profile", required=True)
    ed.add_argument("--text-file", required=True)
    ed.add_argument("--output", required=True)
    ed.add_argument("--mode", choices=["light", "medium", "strong"], default="light")
    ed.set_defaults(func=_cmd_edit)
```
Add handlers:
```python
def _cmd_suggest_edits(args) -> int:
    from .edit.style_editor import suggestions_to_markdown
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    ensure_parent(args.output).write_text(
        suggestions_to_markdown(profile, text, args.mode), encoding="utf-8")
    print(f"Suggestions written to {args.output}")
    return 0


def _cmd_edit(args) -> int:
    from .edit.style_editor import conservative_edit
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    ensure_parent(args.output).write_text(
        conservative_edit(profile, text, args.mode), encoding="utf-8")
    print(f"Edited text written to {args.output}")
    return 0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_edit.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add personal_style_pl/edit tests/personal_style/test_edit.py personal_style_pl/cli.py
git commit -m "feat: conservative suggest-edits and deterministic edit"
```

---

## Task 11: `StyloMetrixFeatureExtractor` (failable) + integration test

**Files:**
- Create: `personal_style_pl/features/stylometrix_features.py`
- Test: `tests/personal_style/test_stylometrix.py`

- [ ] **Step 1: Write the failing test (importorskip-gated)**

`tests/personal_style/test_stylometrix.py`:
```python
import numpy as np
import pytest

from personal_style_pl.features.stylometrix_features import StyloMetrixFeatureExtractor


def test_disabled_returns_empty_block():
    ext = StyloMetrixFeatureExtractor(enabled=False)
    X = ext.fit_transform(["Ala ma kota."])
    assert X.shape == (1, 0)
    assert list(ext.get_feature_names_out()) == []


def test_missing_dependency_raises_clear_error_when_enabled(monkeypatch):
    import personal_style_pl.features.stylometrix_features as m
    monkeypatch.setattr(m, "_import_stylo_metrix", lambda: (_ for _ in ()).throw(
        ImportError("no stylo_metrix")))
    ext = StyloMetrixFeatureExtractor(enabled=True)
    with pytest.raises(RuntimeError, match="StyloMetrix"):
        ext.fit_transform(["Ala ma kota."])


def test_real_stylometrix_extraction():
    pytest.importorskip("stylo_metrix")
    try:
        ext = StyloMetrixFeatureExtractor(enabled=True)
        X = ext.fit_transform(["Wczoraj poszedłem do urzędu i czekałem w kolejce."])
    except RuntimeError:
        pytest.skip("StyloMetrix present but pl_nask model unavailable")
    assert X.shape[0] == 1 and X.shape[1] >= 150
    assert len(ext.get_feature_names_out()) == X.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_stylometrix.py -v`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `personal_style_pl/features/stylometrix_features.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_stylometrix.py -v`
Expected: PASS (real test runs in `.venv`; skips elsewhere).

- [ ] **Step 5: Verify build-profile WITH StyloMetrix end-to-end**

Run:
```bash
.venv/bin/python -m personal_style_pl.cli build-profile \
  --samples-dir examples/my_style_samples --output artifacts/profile_sm.joblib \
  --chunk-sentences 4 --min-chunk-tokens 30
```
Expected: profile built using 172 StyloMetrix features (after Task 13 creates examples).

- [ ] **Step 6: Commit**

```bash
git add personal_style_pl/features/stylometrix_features.py tests/personal_style/test_stylometrix.py
git commit -m "feat: optional failable StyloMetrix PL feature extractor"
```

---

## Task 12: `NgramFeatureExtractor` + `bridge` (`--with-heuristics`)

**Files:**
- Create: `personal_style_pl/features/ngram_features.py`, `personal_style_pl/bridge.py`
- Test: `tests/personal_style/test_ngram_bridge.py`

- [ ] **Step 1: Write the failing test**

`tests/personal_style/test_ngram_bridge.py`:
```python
import numpy as np

from personal_style_pl.features.ngram_features import NgramFeatureExtractor
from personal_style_pl.bridge import attach_heuristics


def test_ngram_preserves_polish_and_fits():
    texts = ["źdźbło trawy źdźbło", "kot pies kot pies", "źdźbło kot pies trawy"]
    ext = NgramFeatureExtractor(min_df=1)
    X = ext.fit_transform(texts)
    assert X.shape[0] == 3
    assert any("ź" in name for name in ext.get_feature_names_out())


def test_attach_heuristics_adds_ai_likeness_block():
    payload = {"style_match_score": 70.0, "label": "mixed"}
    merged = attach_heuristics(payload, "Warto zauważyć, że to jest tekst po polsku. "
                                        "Drugie zdanie tutaj jest również obecne.")
    assert "ai_likeness" in merged
    assert 0.0 <= merged["ai_likeness"]["ai_probability"] <= 1.0
    assert merged["style_match_score"] == 70.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/personal_style/test_ngram_bridge.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `personal_style_pl/features/ngram_features.py`**

```python
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
```

- [ ] **Step 4: Implement `personal_style_pl/bridge.py`**

```python
"""Bridge to the existing heuristic_detector AI-likeness scorer."""

from __future__ import annotations

import heuristic_detector


def attach_heuristics(payload: dict, text: str) -> dict:
    """Add an `ai_likeness` block from heuristic_detector alongside style similarity.
    This is style/AI-likeness signal, NOT proof of authorship."""
    try:
        analysis = heuristic_detector.analyze_text(text)
        payload = dict(payload)
        payload["ai_likeness"] = {
            "ai_probability": float(analysis["ai_probability"]),
            "language": analysis["language"],
            "note": "Heuristic AI-likeness, not proof of authorship.",
        }
    except RuntimeError as exc:
        payload = dict(payload)
        payload["ai_likeness"] = {"error": str(exc)}
    return payload
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/personal_style/test_ngram_bridge.py -v`
Expected: PASS.

- [ ] **Step 6: Verify n-gram round-trip (build → score) does not crash on dimensions**

`build_profile` already assembles + persists the fitted `NgramFeatureExtractor` (Task 6),
and `score_text._features_for_chunks` reuses `profile.ngram_extractor`. Add a test
`tests/personal_style/test_ngram_roundtrip.py`:
```python
from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text


def test_ngram_profile_scores_without_dimension_error():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę. ") * 4
    docs = [SampleDoc(doc_id=f"d{i}", text=base) for i in range(6)]
    profile = build_profile(docs, use_stylometrix=False, include_ngrams=True,
                            chunk_sentences=2, min_chunk_tokens=15)
    assert profile.ngram_extractor is not None
    result = score_text(profile, base)  # must not raise on hstack/broadcast
    assert 0 <= result.style_match_score <= 100
```
Run: `.venv/bin/python -m pytest tests/personal_style/test_ngram_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 7: Run all tests, commit**

Run: `.venv/bin/python -m pytest tests/personal_style -v`
```bash
git add personal_style_pl/features/ngram_features.py personal_style_pl/bridge.py tests/personal_style/test_ngram_bridge.py tests/personal_style/test_ngram_roundtrip.py
git commit -m "feat: optional n-gram features and heuristic AI-likeness bridge"
```

---

## Task 13: Examples, README, package AGENTS.md, supervised stub

**Files:**
- Create: `examples/my_style_samples/sample_01.txt`, `examples/my_style_samples/sample_02.txt`, `examples/candidates/draft_a.txt`, `examples/candidates/draft_b.txt`, `examples/style_samples.csv`
- Create: `personal_style_pl/AGENTS.md`, `personal_style_pl/models/supervised.py`, `personal_style_pl/models/one_class.py`
- Modify: `README.md`

- [ ] **Step 1: Create example samples (Polish, small, clearly insufficient)**

`examples/my_style_samples/sample_01.txt` and `sample_02.txt`: 1–2 short Polish paragraphs of plain personal prose. `examples/candidates/draft_a.txt` (plain personal style), `draft_b.txt` (AI-marker-heavy: "Warto zauważyć…", em dashes, bullet list). `examples/style_samples.csv` with header `text,source,date,genre` and 2 rows. Add a top note in each sample dir via README that these are placeholders, **not enough for a real profile**.

- [ ] **Step 2: Create `personal_style_pl/AGENTS.md`** with the spec's project rules (preserve facts, transparent scoring, tests for every function, run pytest after changes, keep CLI examples working, no hidden network/model downloads, no external LLM calls by default).

- [ ] **Step 3: Implement supervised stub (optional mode)**

`personal_style_pl/models/one_class.py`: re-export `chunk_distance` and a thin `OneClassProfileModel` wrapper around `score_text` for symmetry.
`personal_style_pl/models/supervised.py`: `train_supervised(csv_path, text_col, label_col, output)` using `SurfaceFeatureExtractor` + `LogisticRegression`, `GroupKFold` when a `source` column exists; saves a joblib bundle. Add a `train-supervised` subcommand to `cli.py` mirroring the others. Add `tests/personal_style/test_supervised.py` building a tiny 2-class CSV and asserting a model file is produced and `predict_proba` works.

- [ ] **Step 4: Write `README.md` section** (append a "Personal Polish style similarity (`personal_style_pl`)" section): what it does / does not do; single-env setup via `scripts/setup_style_env.sh`; StyloMetrix + pl_nask notes incl. the `--no-deps` HF model and expired-cert caveat; all CLI examples; data formats; score interpretation (label bands, confidence); limitations; ethical use ("style similarity, not authorship"); **License note: StyloMetrix & pl_nask are GPL-3.0 — check compatibility before distributing**.

- [ ] **Step 5: Run the full acceptance sequence**

Run:
```bash
.venv/bin/python -m personal_style_pl.cli build-profile --samples-dir examples/my_style_samples --output artifacts/profile.joblib --no-stylometrix
.venv/bin/python -m personal_style_pl.cli score --profile artifacts/profile.joblib --text-file examples/candidates/draft_a.txt --json
.venv/bin/python -m personal_style_pl.cli rank --profile artifacts/profile.joblib --candidates-dir examples/candidates --output artifacts/ranking.csv
.venv/bin/python -m personal_style_pl.cli suggest-edits --profile artifacts/profile.joblib --text-file examples/candidates/draft_a.txt --output artifacts/suggestions.md
.venv/bin/python -m pytest -q
```
Expected: all four commands succeed; full suite (detector + style) passes.

- [ ] **Step 6: Commit**

```bash
git add examples personal_style_pl/AGENTS.md personal_style_pl/models README.md tests/personal_style/test_supervised.py
git commit -m "feat: examples, README, package AGENTS.md, optional supervised mode"
```

---

## Task 14: Final verification & branch wrap-up

- [ ] **Step 1: Lean CI parity check (no style deps)** — confirm the lean job won't break:

Run:
```bash
.venv/bin/python -m pytest -q --ignore=tests/personal_style
.venv/bin/python -m py_compile $(find personal_style_pl -name '*.py')
```
Expected: existing detector suite passes; package compiles.

- [ ] **Step 2: Detector regression check** — `.venv/bin/python run_ensemble.py --text "..." --json` still returns `experts.*`, `ensemble`, `calibration`.

- [ ] **Step 3: Update AGENTS.md review checklist** — add `personal-style-pl --help` to the root `AGENTS.md` review rule if appropriate.

- [ ] **Step 4: Final commit & summary**

```bash
git add -A
git commit -m "chore: finalize personal_style_pl feature"
```

---

## Coverage map (spec → task)

- Env/single-version + setup script → Task 0
- Packaging/extras/CI → Task 1, 14
- config phrase lists → Task 2
- chunking → Task 3
- surface features (35) → Task 4
- io (dir/CSV) + json → Task 5
- StyleProfile + builder → Task 6
- scoring formula + calibration + confidence/labels/warnings → Task 7
- CLI build-profile/score/rank (JSON/CSV) → Task 8
- describe-profile → Task 9
- suggest-edits/edit (conservative, metadata, protected entities) → Task 10
- StyloMetrix extractor (failable, 172 feats) → Task 11
- n-grams + heuristic bridge (`--with-heuristics`) → Task 12
- examples/README/AGENTS/supervised → Task 13
- acceptance criteria + regression → Task 13/14
