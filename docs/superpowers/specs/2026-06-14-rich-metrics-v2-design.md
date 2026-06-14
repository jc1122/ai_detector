# Design: Rich metrics v2 — interpretable AI-leaning signals + style correction

**Date:** 2026-06-14
**Status:** Locked (scope approved: full, with Polish-LM perplexity as a gated optional extra)
**Repo:** `ai_detector` (Polish detector + `personal_style_pl`)

## 1. Goal

Add a richer, **interpretable, length-normalized, OOD-aware** metric layer that (a) extends what
we can measure about a text, (b) reads those metrics through an **AI-direction overlay** to surface
literature-backed AI-leaning signals, and (c) feeds the existing `suggest-edits` loop so a draft can
be corrected **toward the user's own human baseline**. Borrow proven libraries; hand-roll only the glue.

This stays **style/marker analysis, not authorship proof.** On Polish/OOD inputs the system must
prefer to **abstain / widen uncertainty** rather than emit a confident AI/human label (false
positives are the dominant harm).

## 2. Evidence base (from web + scite research)

- Fine-tuned transformer detectors collapse under domain shift; an **interpretable stylometric
  hybrid matches them in-domain**, with **sentence-level perplexity coefficient-of-variation** and
  **AI-phrase density** as the most discriminative features (Baidya et al., 2026,
  https://doi.org/10.48550/arxiv.2603.17522). Modern LLM text is **lower-perplexity** than human
  (polarity inversion).
- Repeated higher-order n-grams over-appear in machine text (Gallé et al., 2021,
  https://doi.org/10.48550/arxiv.2111.02878).
- Statistics/probability features generalize OOD better than fine-tuned PLMs (Chen & Liu, 2023,
  https://doi.org/10.1007/978-981-99-4752-2_60).
- Multilingual detection: Polish poorly covered, **false positives are the key harm** → abstain on
  OOD (Macko et al., 2023, https://doi.org/10.18653/v1/2023.emnlp-main.616).
- Length-robust lexical diversity: prefer **MATTR/MTLD** over raw TTR (Bestgen, 2024,
  https://doi.org/10.2139/ssrn.4928392).
- Probability-curvature zero-shot detection: **Fast-DetectGPT** (single model) — method we adapt
  with a Polish LM.

## 3. Libraries to borrow (verified compatible with the 3.12 env)

| Capability | Library | Compatibility |
|---|---|---|
| MATTR, MTLD, HD-D, RTTR/CTTR | **LexicalRichness 0.5.1** (pure-Python) | ✅ no pin conflicts |
| Dependency distance, POS proportions, descriptive stats (NOT entropy/perplexity — NaN for PL) | **TextDescriptives 2.8.4** (spaCy v3 component) | ✅ verified: adds with no spaCy/numpy change; reuses `pl_nask` |
| Polish LM perplexity / curvature | **papuGaPT2** (`dkleczek/papuGaPT2`, ~500 MB) via existing `torch`+`transformers` | ✅ no new pins; model downloaded by a setup step (like `pl_nask`) |
| Fast-DetectGPT curvature method | adapt the algorithm (one model) | in-house, uses papuGaPT2 logits |

New extras: `rich = ["lexicalrichness", "textdescriptives"]`; `rich-perplexity = []` (relies on
existing torch/transformers; papuGaPT2 fetched by `scripts/setup_style_env.sh` addition).

## 4. Components

### 4.1 `personal_style_pl/features/rich_metrics.py` (always-on, cheap)
`RichMetricsExtractor` + `rich_metrics_for_text(text)` returning a stable-named dict:
- **Lexical diversity** (LexicalRichness): `mattr` (window 50), `mtld`, `hdd`, `rttr`, `cttr`.
- **Burstiness**: `sentence_len_cv` (reuse), `burstiness_coeff` = (σ−μ)/(σ+μ) on sentence-token lengths.
- **Repetition**: `repeated_4gram_ratio` (share of token 4-grams that recur) — Gallé-style signal
  beyond the existing max bi/trigram.
- **Length-normalized densities** (per 1,000 tokens): `em_dash_per_1k`, `ai_phrase_per_1k`,
  `boilerplate_per_1k`, `transition_per_1k`, `hedge_per_1k` — fixes the length confound.

### 4.2 `personal_style_pl/features/textdescriptives_features.py` (gated; uses spaCy)
Wrap TextDescriptives over the `pl_nask` spaCy pipeline. **Verified non-NaN columns only**:
`dependency_distance_mean/std`, `prop_adjacent_dependency_relation_mean/std`, `pos_prop_*` (UD tags),
`syllables_per_token_mean/std`, `token_length_mean/std`.
- **entropy/perplexity are EXCLUDED**: TextDescriptives' information_theory needs a spaCy
  lexeme-probability table that does not exist for Polish → NaN (verified audit 2026-06-14). Real
  perplexity/entropy come from §4.3 (papuGaPT2).
- Readability indices are English-calibrated → excluded from the primary set.
- Model-gated: skips cleanly without `pl_nask`.

### 4.3 `personal_style_pl/features/perplexity_features.py` (GATED optional) — VERIFIED
papuGaPT2-based (loads ~21 s; verified working), lazy import + clear `RuntimeError` if missing:
- `lm_perplexity` (**robust: median** per-sentence PPL), `lm_logprob_mean`,
- `sent_perplexity_cv` (**robust: IQR/median** per-sentence PPL dispersion),
- `fastdetect_curvature` (std of per-sentence log-PPL; single-model proxy).
- Polarity inversion: lower PPL / lower dispersion ⇒ more AI-leaning.
- **Robust aggregation required** — per-sentence PPL has heavy outliers from PDF-extraction noise.
- **Honest caveat (verified):** on the Polish chemistry domain perplexity did NOT cleanly separate
  human (median≈50) from AI-assisted (median≈54) — ADVISORY signal, not a verdict.
- Exposed as `PerplexityExtractor` (+ `PERPLEXITY_FEATURE_NAMES`) so it can be wired into the profile
  baseline (opt-in `--with-perplexity`), like StyloMetrix/n-gram.

### 4.4 `personal_style_pl/ai_markers.py` (the overlay — the interpretive core)
- `AI_MARKERS`: a curated table mapping marker features → AI direction (sign) + literature source +
  human-readable rationale + a `suggestion` template. Covers: low burstiness/`mattr`, perplexity
  inversion + low `sent_perplexity_cv`, high `em_dash_per_1k`, high `boilerplate_per_1k`/
  `transition_per_1k`, high `repeated_4gram_ratio`.
- `ai_marker_report(text, profile=None, with_perplexity=False)` →
  per-marker {value, ai_direction, your_baseline (if profile), deviation, leaning ∈ matches/AI-leaning},
  a transparent aggregate **`ai_leaning_score` (0–100, explained, never a hard label)**, an
  **`ood`/`confidence`** block (language via `heuristic_detector.detect_language`; abstain on PL), and
  warnings. Always carries "marker-based heuristic, not authorship proof."

### 4.5 CLI
- `personal-style-pl ai-markers --text-file X [--profile P] [--with-perplexity] [--json]` → overlay report.
- `suggest-edits` consumes the overlay: concrete edits ("raise sentence-length variation toward your
  CV≈…", "cut em-dashes toward your ~N/1k", "reduce 'warto zauważyć'-type phrases").
- Detector: `ai-detector-heuristic --rich` adds the rich-metric block + `ood`/abstain flag to output
  (additive; existing keys unchanged).

### 4.6 Interpretable hybrid (optional)
Extend `train-supervised` to accept `--features rich` → train LogisticRegression (default) / XGBoost
(if installed) on the rich vector; explanations via coefficients (logistic) / permutation importance.

## 5. Principles / caveats (encoded)
- Transparent + interpretable; **length-normalized** everywhere; perplexity polarity correct.
- **Abstain on Polish/OOD**; never output a confident AI/human label for PL.
- Readability indices are English-calibrated → relative-only for PL.
- Reuse existing extractors/tokenizer/phrase lists; DRY. Gated heavy pieces degrade gracefully.

## 6. Dependencies / CI
- `[rich]` extra (`lexicalrichness`, `textdescriptives`); `[rich-perplexity]` relies on existing
  torch/transformers + the papuGaPT2 model fetched by `setup_style_env.sh`.
- Light `style-test` CI installs only `lexicalrichness` (covers `rich_metrics`/`ai_markers` tests).
  TextDescriptives **and** perplexity tests are **model-gated** (skip without a real spaCy model /
  cached papuGaPT2) — do NOT add `textdescriptives` to light CI (it pulls a model-less spaCy).
- Lean detector CI untouched.

## 7. Build order
0. Deps + setup script (incl. papuGaPT2 fetch, verified) + CI (lexicalrichness only) + branch.
1. `rich_metrics.py` (lexical diversity + burstiness + ngram + densities) + tests.
1b. Wire rich (default) + perplexity (opt-in) into `build_profile` + `similarity` baseline + tests.
2. `textdescriptives_features.py` (dependency/POS/descriptive — NO entropy/perplexity) + model-gated tests.
3. `ai_markers.py` overlay (+ OOD/abstain) + tests.
4. CLI `ai-markers` + `suggest-edits` integration + tests.
5. `perplexity_features.py` (gated, robust median/IQR) + model-gated tests.
6. `ai-detector-heuristic --rich` integration + tests.
7. Hybrid `train-supervised --features rich`.
8. README + `personal_style_pl/AGENTS.md` update + acceptance run (incl. `--with-perplexity`).

## 8. Acceptance criteria
- `personal-style-pl ai-markers --text-file examples/candidates/draft_b.txt --json` returns rich
  metrics + `ai_leaning_score` + `ood`/confidence + per-marker explanations, no crash, no network.
- Length-normalized densities reproduce the Chemik-vs-human em-dash contrast.
- TextDescriptives + perplexity paths skip cleanly when deps/model absent.
- Lean + style CI green; detector entrypoints unchanged.
- Full suite passes; README documents setup, commands, and the "not authorship proof / abstain on PL" stance.

## 9. References
- Baidya, M. S., Baidya, S. S., & Chawla, C. (2026). arXiv. https://doi.org/10.48550/arxiv.2603.17522
- Bestgen, Y. (2024). SSRN/Elsevier. https://doi.org/10.2139/ssrn.4928392
- Chen, Z., & Liu, H. (2023). Springer. https://doi.org/10.1007/978-981-99-4752-2_60
- Gallé, M., Rozen, J., & Kruszewski, G. (2021). arXiv. https://doi.org/10.48550/arxiv.2111.02878
- Macko, D., Móro, R., Uchendu, A., et al. (2023). EMNLP. https://doi.org/10.18653/v1/2023.emnlp-main.616
- Bao, G., et al. (2024). Fast-DetectGPT. ICLR. (repo: github.com/baoguangsheng/fast-detect-gpt)
- Hans, A., et al. (2024). Binoculars. ICML. https://doi.org/10.48550/arxiv.2401.12070
