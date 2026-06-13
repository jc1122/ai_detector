# Design: `personal_style_pl` — Polish personal writing-style similarity

**Date:** 2026-06-13
**Status:** Locked (design approved)
**Repo:** `ai_detector` (existing Polish/EN AI-text detector)

## 1. Goal

A local CLI/package that lets the user (1) ingest their own Polish writing samples,
(2) build a personal style profile, (3) score how much a new Polish text resembles
their style, (4) rank candidate drafts, (5) produce an interpretable report of which
features match/diverge, and (6) optionally suggest conservative, transparent edits.

This is **style similarity**, never "authorship verification" or "proof of AI authorship".

## 2. Non-goals

- No external LLM API calls in the default implementation.
- No hidden network calls or runtime model downloads (model/env setup is explicit and scripted).
- No detector-evasion loop. Edit mode is conservative, deterministic, and meaning-preserving.

## 3. Environment (decided + verified)

**The whole project standardizes on a single Python 3.12 venv.** No dual-version setup.
Python 3.14 (the previous `.venv`) is dropped because StyloMetrix's hard-pinned
`spacy==3.7.2` stack has no wheels above cp312.

Verified facts (live-tested 2026-06-13):

- All `pl_nask` model variants (0.0.5/0.0.7/large/nomorf) are HerBERT-**transformer**-based,
  so `spacy-transformers` → `torch` + `transformers<4.50` are unavoidable.
- Working, verified version set in one 3.12 env:
  `numpy 1.26.4 (<2)`, `pandas 2.1.4`, `scikit-learn 1.9.0`, `scipy 1.17.1`, `joblib 1.5.3`,
  `regex 2026.5.9`, `rich 15.0.0`, `spacy 3.7.2`, `spacy-transformers 1.3.9`, `thinc 8.2.4`,
  `stylo_metrix 0.1.9.1`, `transformers 4.49.0`, `torch 2.10.0`, `morfeusz2`, `pl_nask 0.0.7`.
- **`numpy<2` is mandatory** (thinc 8.2.4 was built against numpy 1.x; numpy 2 → ABI crash).
- **`pl_nask` must be installed `--no-deps`** from the TLS-valid HF mirror
  `https://huggingface.co/ipipan/pl_nask/resolve/main/pl_nask-0.0.7.tar.gz`
  (it declares `spacy<3.6`, which would otherwise drag an unbuildable `thinc 8.1.x`;
  the official IPI PAN host has an expired TLS cert).
- **Prior functionality verified compatible**: 95/95 existing detector tests pass and real
  MELD+TMR+RAID ensemble inference produces correct calibrated scores on this stack.
  spaCy emits a harmless `W095` (model trained on spaCy 3.5.0).
- The env is provisioned with `uv` (user-space, no root) → `scripts/setup_style_env.sh`.

### CLI framework decision

**`argparse` (stdlib) + `rich` for tables — NOT `typer`.** Verified: in this env `typer`
is capped at `0.9.4` (by `spacy`'s `typer<0.10`), which is **broken** against the
co-installed `click 8.4.1` (option parsing fails). argparse matches the 5 existing
detector CLIs, has zero version sensitivity, and removes `typer`+`click` from the new
package's risk surface. `rich` is used only for optional pretty tables with a plain-text fallback.

## 4. Package layout

```
personal_style_pl/
  __init__.py
  cli.py                 # argparse dispatch; `python -m personal_style_pl.cli`
  config.py              # phrase lists (PL transitions/boilerplate/hedges), thresholds
  io.py                  # read samples dir / CSV; write json/csv/md; artifact paths
  textsplit.py           # sentence-aware chunking (8 sentences, min 120 tokens, doc ids)
  bridge.py              # optional heuristic_detector AI-likeness bridge
  features/
    __init__.py
    surface_features.py      # SurfaceFeatureExtractor (reuses heuristic_detector primitives)
    stylometrix_features.py  # StyloMetrixFeatureExtractor (lazy import, failable)
    ngram_features.py        # NgramFeatureExtractor (optional, --include-ngrams)
    combined_features.py     # CombinedFeatureExtractor
  profile/
    __init__.py
    build_profile.py     # StyleProfile dataclass + builder
    similarity.py        # one-class scoring
    calibration.py       # temperature calibration from self-distances
    report.py            # ScoreResult assembly + describe-profile
  edit/
    __init__.py
    style_editor.py      # StyleSuggestionEngine + conservative deterministic edit
    rules.py             # edit/suggestion rules
  models/
    __init__.py
    supervised.py        # optional LogisticRegression/RandomForest/LightGBM
    one_class.py         # one-class profile model helpers
  utils/
    __init__.py
    logging.py
    json.py
  AGENTS.md              # project rules for the new feature (separate from root AGENTS.md)
tests/personal_style/
  test_surface_features.py
  test_chunking.py
  test_profile_smoke.py
  test_cli_smoke.py
  test_stylometrix.py    # importorskip-gated
examples/
  my_style_samples/{sample_01.txt,sample_02.txt}
  candidates/{draft_a.txt,draft_b.txt}
  style_samples.csv
scripts/setup_style_env.sh
```

## 5. Integration with existing heuristics (reuse + bridge)

- `surface_features.py` imports `_extract_words`, `_split_sentences`, `_split_paragraphs`,
  `_fold_text` from `heuristic_detector` — single source of truth for Polish tokenization
  (diacritics preserved: ą ć ę ł ń ó ś ź ż).
- `config.py` phrase lists seed from `heuristic_detector.AI_PHRASES_PL` / `AI_WORDS_PL`
  plus the spec's transition/boilerplate/hedge lists, deduped and diacritic-folded.
  These are **style indicators, not proof of AI authorship**.
- `bridge.py`: `score`/`rank` accept `--with-heuristics` to attach the existing AI-likeness
  block (`heuristic_detector.analyze_text`) alongside the style-similarity result, so one
  command reports "X% like my style AND Y% AI-likeness on the heuristics".

## 6. Feature extraction

All extractors are sklearn-compatible (`BaseEstimator`, `TransformerMixin`), return a
pandas DataFrame or numpy array with **stable feature names** via `get_feature_names_out`.

- **SurfaceFeatureExtractor** — deterministic, Polish-aware, ~35 features:
  char/token/sentence/paragraph counts; avg/median/std/min/max sentence length; sentence_len_cv;
  avg paragraph length; avg token length; type_token_ratio; hapax_ratio; punctuation_density;
  comma/semicolon/colon/dash per sentence; question/exclamation counts; parenthesis/quote/digit
  density; uppercase_ratio; newline_density; bullet_like_line_ratio; repeated_bigram/trigram_ratio;
  top_token_repetition_ratio; first_person_singular/plural counts; hedge_count;
  transition_phrase_count; generic_boilerplate_count; average_clause_marker_count.
- **StyloMetrixFeatureExtractor** — `sm.StyloMetrix("pl")`, **172 PL features** (verified).
  Lazy import; raises a clear `RuntimeError` if `stylo_metrix`/model missing, **unless `enabled=False`**.
- **NgramFeatureExtractor** — char `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2)`
  and word `(1,2), min_df=2`. Optional via `--include-ngrams`; off by default (overfits to topic).
- **CombinedFeatureExtractor** — composes the above with a saved feature mask.

## 7. Chunking (`textsplit.py`)

- Chunk by sentences (default 8), minimum 120 tokens; keep document IDs to avoid leakage.
- Very short corpus warning: "Profile is weak: provide at least 10–20 writing samples or 5,000+ words."

## 8. Profile model + scoring

`StyleProfile` dataclass: `profile_id, created_at, language="pl", feature_names, center,
scale, robust_center, robust_scale, covariance_model?, training_scores,
training_sample_count, training_chunk_count, total_tokens, genres, config,
feature_extractor, similarity_calibrator, warnings`. Persisted via `joblib`.

One-class scoring:

1. Split candidate into chunks; extract features; standardize with profile center/scale (RobustScaler-style).
2. Robust z: `z = clip(|(x - center) / scale|, max=8)`; `distance = median(z over stable features)`.
3. `raw = 100 * exp(-distance / temperature)`, temperature calibrated from training self-distances
   when ≥ enough chunks. Also compute cosine similarity to centroid.
4. `final = 0.7 * z_component + 0.3 * cosine_component`. Optional Mahalanobis when enough chunks.
5. Aggregate chunk scores: median with a lower-quartile penalty.
6. Labels: 80–100 `close_to_my_style`; 55–79 `mixed`; 0–54 `far_from_my_style`;
   `insufficient_text` when too few tokens.

Feature stability: exclude near-zero-variance features and count features unreliable for short
chunks; save the selected feature mask in the profile.

Confidence: `high` (chunks ≥30 and candidate tokens ≥300); `medium` (chunks ≥15 and tokens ≥200);
`low` otherwise. With <10 training chunks use only conservative metrics and mark confidence low.

Warnings: very short candidate; small profile; genre mismatch; StyloMetrix unavailable
(surface only); "This is style similarity, not proof of authorship."

`ScoreResult` dataclass per spec: `style_match_score, label, confidence, warnings, summary,
top_matches, top_mismatches, chunk_scores`.

## 9. CLI (argparse + rich)

Subcommands with the spec's exact flags and output contracts:
`build-profile` (`--samples-dir`|`--csv` `--text-col` `--output` `--report` `--min-chunk-tokens`
`--chunk-sentences` `--include-ngrams` `--no-stylometrix`), `score` (`--profile` `--text-file`
`--json` `--with-heuristics`), `rank` (`--profile` `--candidates-dir` `--output` CSV columns
`filename,style_match_score,label,confidence,word_count,warnings`), `describe-profile`
(`--profile` `--output` markdown), `suggest-edits` (`--profile` `--text-file` `--output` `--mode`),
`edit` (`--profile` `--text-file` `--output` `--mode`), `train-supervised`
(`--csv` `--text-col` `--label-col` `--output`). Pretty rich tables by default; `--json` where specified.

## 10. Edit / suggest-edits

`StyleSuggestionEngine` consumes candidate text + profile + feature report and emits markdown
grouped by issue (overall score, top 5 divergences, suggestions, examples, warnings) with
metadata `machine_assisted_style_edit: true`, `profile_used`, `mode: light|medium|strong`.
Suggestions: split overlong / merge choppy sentences; review comma rhythm; trim boilerplate;
match first-person/transition density when context allows; reduce repetition without inventing facts;
convert lists↔prose toward profile.

`edit` (deterministic, conservative): normalize whitespace; split overlong sentences only at safe
punctuation; remove diverging boilerplate; adjust transition density toward profile; flag vague
text rather than inventing specifics. **Never change** names, dates, numbers, legal/medical/financial
claims, citations, quotes, URLs, code. `EditSuggestion` dataclass: `issue, reason, suggestion,
before?, after?, severity`.

## 11. Optional supervised mode

If contrast CSV (`label ∈ {mine, other}`) is supplied: `LogisticRegression` default,
`RandomForest` optional, `LightGBM` if installed; `GroupKFold` on source/document IDs.
Explanations: one-class → rank by |z|; logistic → coefficient×value; trees → permutation importance
(SHAP optional only if installed).

## 12. Packaging & dependencies

- New package added to setuptools: `[tool.setuptools] packages = ["personal_style_pl", ...subpackages]`
  (kept alongside existing `py-modules`). Console script `personal-style-pl = personal_style_pl.cli:main`;
  `python -m personal_style_pl.cli` also works.
- `requires-python = ">=3.10"` — **unchanged / not narrowed** (the detector runs broadly;
  3.12 is enforced for the unified/StyloMetrix env via the setup script + README, not metadata).
- Extras:
  - `style = ["numpy<2","pandas","scikit-learn","scipy","joblib","rich","regex"]`
  - `style-stylometrix = ["stylo_metrix","spacy==3.7.2","spacy-transformers>=1.3,<1.4","numpy<2"]`
    — **NOTE: this extra alone does NOT yield working StyloMetrix**; `pl_nask` must be installed
    `--no-deps` via `scripts/setup_style_env.sh`. Documented loudly.
  - `style-ml = ["lightgbm","shap"]`
- Base detector deps (`torch`, `transformers`, `safetensors`) stay **unpinned** so CI runtime-smoke
  keeps testing transformers 5.9; the 4.49 cap only applies when the style extra is co-installed.
- `MANIFEST.in`: add `recursive-include examples *`. `.gitignore`: `artifacts/` (+ `.venv-style/` legacy) added.
- `scripts/setup_style_env.sh`: reproducible env build (uv 3.12 → install `.[style,style-stylometrix,test]`
  with `numpy<2` → install `pl_nask` `--no-deps` from HF). Exact verified recipe.

## 13. CI/CD changes

- **Lean detector job** (`ci.yml`): change `python -m pytest -q` → `python -m pytest -q --ignore=tests/personal_style`
  (otherwise it collects the new tests with no sklearn and fails at collection). `py_compile` glob unaffected.
- **New style job** (`ci.yml`): install `.[style,test]` (light: sklearn/pandas/scipy/numpy<2, **no torch/stylometrix**);
  run `pytest -q tests/personal_style`, `python -m personal_style_pl.cli --help`, and `py_compile` over the package.
- **runtime-smoke / release**: unchanged.
- CI Python stays 3.11 (in range).

## 14. Tests (pytest-style functions; coexist with existing unittest)

Diacritic preservation; surface feature stable names; empty/short text handled gracefully;
chunking; build-profile smoke (`--no-stylometrix`); score returns JSON; rank outputs CSV;
suggest-edits outputs markdown; StyloMetrix integration test `pytest.importorskip`-gated.

## 15. Risks & gaps (accepted)

1. **CI does not exercise real StyloMetrix** (torch + 474 MB model impractical in CI); the
   StyloMetrix test skips in CI and is verified only locally.
2. **`.[style-stylometrix]` ≠ working StyloMetrix** — `pl_nask` needs the setup script.
3. **Stack divergence**: local `transformers 4.49` vs CI/old `5.9`; detector verified on both.
4. **No `LICENSE` file**; `stylo_metrix`/`pl_nask` are **GPL-3.0**. Acceptable as an optional
   extra for personal use; README must document GPL implications; distributing the whole would
   trigger GPL obligations.
5. **API discipline**: new code targets `numpy 1.26` / `pandas 2.1` APIs.

## 16. Build order

0. Env consolidation: `scripts/setup_style_env.sh` rebuilds a single `.venv` on 3.12; verify detector + StyloMetrix.
1. `config` + `io` + `textsplit` + `surface_features` (+ tests).
2. `build_profile` + `StyleProfile` + `similarity` + `calibration`.
3. `score` + `rank` CLI + `report` (ScoreResult).
4. `describe-profile`.
5. `suggest-edits` + `edit` (conservative/deterministic).
6. `StyloMetrixFeatureExtractor` + `NgramFeatureExtractor` + `bridge` (`--with-heuristics`).
7. Optional supervised mode.
8. README + `personal_style_pl/AGENTS.md` + pyproject/CI/MANIFEST wiring.

## 17. Acceptance criteria

- `build-profile --samples-dir examples/my_style_samples --output artifacts/profile.joblib --no-stylometrix` works.
- `score --profile artifacts/profile.joblib --text-file examples/candidates/draft_a.txt --json` works.
- `rank --profile artifacts/profile.joblib --candidates-dir examples/candidates --output artifacts/ranking.csv` works.
- `suggest-edits --profile artifacts/profile.joblib --text-file examples/candidates/draft_a.txt --output artifacts/suggestions.md` works.
- StyloMetrix path works in the unified 3.12 env (172 features).
- pytest passes (lean + style jobs); README complete; no hidden internet/LLM calls.
