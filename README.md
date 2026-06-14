# AI Detector CLI

This repo provides:
- `ai-detector`: 3-expert ensemble classification with MELD, TMR, and MAGE
  (ModernBERT from `raid_model`).
- `ai-detector-heuristic`: fast local rule-based scoring for triage before
  running the heavier model ensemble.
- `ai-detector-calibrate`: grid-search operating-point calibration from stored
  evaluation scores.
- `personal-style-pl`: Polish personal writing-style similarity + an interpretable,
  OOD-aware **AI-leaning** overlay (`ai-markers`).

## Quick start (TL;DR)

Full env (detector + style + the papuGaPT2 perplexity model): `./scripts/setup_style_env.sh`.
Detector only: `python -m pip install -e .`.

```bash
# A. Fast AI-text triage (no model weights). --rich adds the interpretable
#    rich-metric + AI-leaning block (needs the style extras).
ai-detector-heuristic --text-file examples/candidates/draft_b.txt --json --rich

# B. "Is this AI, or my own human style?" — score a candidate against a KNOWN-HUMAN
#    baseline you build once from reference samples (a folder of .txt files):
personal-style-pl build-profile --samples-dir examples/my_style_samples \
  --output artifacts/human.joblib --no-stylometrix
personal-style-pl ai-markers --text-file examples/candidates/draft_b.txt \
  --profile artifacts/human.joblib --json
#    (append --with-perplexity to BOTH commands for the papuGaPT2 perplexity signal)
```

Read `ai_leaning_score` (0–100, vs your baseline), each marker's `leaning` + `counted`, and
(with perplexity) `perplexity_flag`. **Advisory triage, not proof of authorship; it abstains on
Polish** (`ood_or_unreliable: true`, low confidence). Details: detector in §1–11, style +
AI-vs-human in §12 ("Judging AI-generated vs human text").

## CLI install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Then use the installed entrypoints:

```bash
ai-detector --help
ai-detector-heuristic --help
ai-detector-calibrate --help
ai-detector-deploy --help
```

On externally managed Python installs, keep the editable install inside a virtual
environment. For an existing environment where dependencies are already present,
use `.venv/bin/python -m pip install -e . --no-deps`.

## 1) Ensemble review summary

- Current contract: `meld`, `tmr`, `raid` experts are combined as a weighted
  ensemble from active experts only.
- `--weights` accepts `meld,tmr` (mapped to `meld,tmr,0`) or `meld,tmr,raid`.
- In local practice, `tmr`-only runs are the reference for Polish/OOD quick checks
  because sample outputs are stored in `data/evaluation/*`.
- The default runtime profile is `pl-technical-ood`, a packaged hybrid
  operating point for this PL/OOD use case. It uses model weights
  `meld,tmr,raid = 0.75,0.00,0.25`, blends in the fast heuristic with
  `heuristic_weight = 0.60`, and sets `threshold = 0.434552`.
- Use `--profile legacy-default` to run the old raw default weights
  `0.34,0.33,0.33`.
- TMR and RAID catch synthetic AI controls, but both over-call many Polish
  pre-2020 technical papers at the default `0.5` threshold; do not use TMR-only
  as a reliability proxy for Polish OOD text.
- The heuristic CLI is separate and model-free. Use it for cheap triage; use
  `ai-detector` for detailed model-based analysis.

## 2) Model folders and deployment

Runtime model weights are large Hugging Face artifacts, so they are not vendored
inside the Python wheel. The package declares the code dependencies in
`pyproject.toml`; `ai-detector-deploy` materializes the model data dependency
into the local runtime folders.

Fresh checkout deployment:

```bash
ai-detector-deploy --all
```

This downloads the packaged model set:

| expert | Hugging Face model | local folder |
| --- | --- | --- |
| MELD | `anon-review-meld-2026/meld` | `./meld_model` |
| TMR | `Oxidane/tmr-ai-text-detector` | `./tmr_model` |
| MAGE/RAID | `GeorgeDrayson/modernbert-ai-detection-raid-mage` | `./raid_model` |

List the packaged sources without downloading:

```bash
ai-detector-deploy --list-models
```

For reproducible deployment, pin immutable revisions per model:

```bash
ai-detector-deploy \
  --all \
  --meld-revision <immutable_meld_commit_or_tag> \
  --tmr-revision <immutable_tmr_commit_or_tag> \
  --raid-revision <immutable_raid_commit_or_tag>
```

You can still pull a single custom Hugging Face model into a local folder:

```bash
ai-detector-deploy \
  --model-id anon-review-meld-2026/meld \
  --revision <immutable_meld_commit_or_tag> \
  --target-dir ./meld_model
ai-detector-deploy \
  --model-id Oxidane/tmr-ai-text-detector \
  --revision <immutable_tmr_commit_or_tag> \
  --target-dir ./tmr_model
ai-detector-deploy \
  --model-id GeorgeDrayson/modernbert-ai-detection-raid-mage \
  --revision <immutable_raid_commit_or_tag> \
  --target-dir ./raid_model
```

Defaults and behavior:
- `ai-detector-deploy` default target is `./meld_model` (override with `--target-dir`).
- `--revision` defaults to `main`; with `--all`, it is the shared fallback
  revision unless `--meld-revision`, `--tmr-revision`, or `--raid-revision` is
  provided. Treat `main` as a mutable pointer.
  For reproducible deployments, use an immutable revision (commit SHA or tag) from
  the model page under Commits/Versions.
- Re-running the same command is idempotent; unchanged files are skipped.
- A manifest is written as `ai_detector_model_manifest.json` in each target folder.
  It stores deployment metadata including `model_id`, the requested `revision`,
  `fetched_at`, and other model metadata returned by Hugging Face. If you pass
  `main`, the manifest records `main`; use an immutable commit SHA or tag when
  the manifest must identify a reproducible model version directly.

Expected local directories for inference:
- `./meld_model`
- `./tmr_model`
- `./raid_model`

Checkout fallback: use `python3 deploy_meld.py ...` instead of
`ai-detector-deploy ...` when the package has not been installed yet.

MELD note: `./meld_model` contains the MELD head and tokenizer. Its
`meld_config.json` points the backbone to `jhu-clsp/ettin-encoder-400m`, so
`--local-files-only` also requires that backbone to be present in the local
Hugging Face cache. Run once online without `--local-files-only`, or pre-cache
that backbone, before expecting fully offline MELD inference.

## 3) Run ensemble inference

```bash
# default profile: pl-technical-ood
ai-detector --text "The sky is blue and clear."

# input from stdin (pipe)
printf "The sky is blue and clear." | ai-detector --json

# input from file
ai-detector --text-file ./input.txt --json

# explicit weights and chunk controls
ai-detector \
  --weights 0.34,0.33,0.33 \
  --max-chunks 4 \
  --batch-size 4 \
  --overlap 64 \
  --text "The sky is blue and clear."

# zero-weight expert skip
ai-detector --weights 1.0,0.0,0.0 --text "The sky is blue and clear." --json

# explicit Polish/OOD profile
ai-detector \
  --profile pl-technical-ood \
  --text-file ./input.txt \
  --json

# legacy raw default weights
ai-detector --profile legacy-default --weights 0.34,0.33,0.33 --text-file ./input.txt --json
```

Notes:
- `--weights` accepts two or three values: `meld,tmr,(raid)`.
- `--batch-size` controls chunk batch size for scoring.
- `--max-chunks` caps the number of chunks scored per expert.
- `--threshold` compares only the final ensemble value.
- `--profile` selects a packaged operating profile; the default is
  `pl-technical-ood`.
- `--calibration-file` supplies default weights, threshold, and optional
  heuristic blend weight from an external JSON operating-point calibration
  unless the corresponding CLI options are explicitly passed.
- `--heuristic-weight` blends the heuristic score into the final ensemble; the
  PL/OOD profile sets this automatically.
- `--text`, `--text-file`, stdin pipe, and `--json` output mode are supported.
- `--quiet` suppresses third-party stderr chatter during model loading/scoring;
  user-facing `Error: ...` messages are still printed on failure.

## 4) Fast heuristic triage

The heuristic path is a local Python port of the reviewed browser-side detector
style: sentence variation, entropy, n-gram repetition, punctuation/list markers,
Polish/English AI phrase markers, and AI-style word stems. It does not load model
weights and is intended as a red-flag signal only.

```bash
# direct text
ai-detector-heuristic --text "Warto zauważyć, że to krótki test operatora z pełnym zdaniem."

# stdin pipe
printf "Warto zauważyć, że w dzisiejszym świecie AI odgrywa kluczową rolę.\n" \
  | ai-detector-heuristic --json

# file input
ai-detector-heuristic --text-file ./input.txt --json

# optional: attach the interpretable rich-metric block + AI-leaning overlay
# (lazy-imports personal_style_pl; the base path keeps zero extra dependencies)
ai-detector-heuristic --text-file ./input.txt --json --rich
```

With `--rich`, the payload gains `experts.heuristic.rich_metrics` (MATTR/MTLD,
burstiness, length-normalized phrase densities) and `experts.heuristic.ai_leaning`
(the AI-direction overlay; abstains on Polish). See §12 for the metrics and the
interpretation/calibration. Without `--rich` the lean detector is unchanged.

Heuristic JSON uses the same top-level operator sections:
- `experts.heuristic.ai_probability`, `human_probability`, `site_metrics`, `categories`, `signals`, `metrics`
- `ensemble.ai_probability`, `ensemble.human_probability`, `ensemble.threshold`, `ensemble.label`
- `calibration.status = uncalibrated_heuristic`

Operator fields:
- `experts.heuristic.site_metrics.*.ai_probability_percent` mirrors the site UI
  category bars: `Zmienność tekstu`, `Słownictwo`, `Entropia`,
  `Powtarzalność`, `Sygnatury AI`, `Struktura`.
- `experts.heuristic.categories.*.score` is the internal human-like score used
  to compute those bars; the site-style AI percentage is `100 - score`.
- `experts.heuristic.signals` lists detected AI phrases, AI-style words, and
  em/en dash count for quick review.

Parity smoke against the reviewed web-detector behavior:
- Polish marker-heavy sample: `ai_probability = 0.99`
- plain Polish narrative sample: `ai_probability = 0.29`
- English marker-heavy sample: `ai_probability = 0.99`
- very short text is rejected before scoring

These are raw uncalibrated rule scores, not proof of authorship.

The initial six-sample smoke showed moderate heuristic-vs-ensemble agreement
(`Pearson=0.690`, `Spearman=0.771`), but the broader pre-2020 Polish technical
paper set showed weak or negative correlation with model scores. Treat the
heuristic as a separate triage signal, not a model-score substitute.

## 5) Output contract (AI / human)

JSON fields used by operators:
- `weights`
- `experts.meld`, `experts.tmr`, `experts.raid` each with:
  - `ai_score`, `human_score`, `ai_probability`, `human_probability` when loaded
  - `loaded`, `chunks`, optional `notes`
- `ensemble.ai_probability`, `ensemble.human_probability`, `ensemble.threshold`, `ensemble.label`
- `calibration.status`, `calibration.calibrated`, `calibration.message`

Decision logic:
- `ensemble.model_ai_probability` is the weighted average of loaded model
  experts.
- `ensemble.ai_probability` is `ensemble.model_ai_probability` unless a
  non-zero `heuristic_weight` is active; with heuristic blending it is
  `(1 - heuristic_weight) * model_ai_probability + heuristic_weight * heuristic_ai_probability`.
- `ensemble.label` is `ai` if `ensemble.ai_probability >= threshold`, else `human`.
- `ensemble.human_probability` is `1 - ensemble.ai_probability`.
- `weights` reports effective final weights. With the default PL/OOD profile
  this is `meld=0.30`, `tmr=0.00`, `raid=0.10`, `heuristic=0.60`.

Scoring details:
- For loaded experts, `human_probability = 1 - ai_probability`.
- For skipped experts (`loaded: false`, typically due to zero weight), all score/probability
  fields are `null` and `notes` explains the skip.
- Without calibration configuration, all probabilities are **raw uncalibrated scores**.
- With `--profile pl-technical-ood` or `--calibration-file`,
  `calibration.status` becomes `operating_point_calibrated`. This means
  weights/threshold were selected from local validation data; individual expert
  probabilities are still raw scores, not probability-calibrated estimates.

Privacy note:
- `text_preview` in JSON output is the first 250 characters of raw input text.
- This preview is emitted as part of CLI output and can be copied into logs by operators;
  avoid sending sensitive or regulated text through the same output channel unless
  log redaction is in place.
- Keep `text_preview` as a debug aid only; it is not a sanitized/safe-to-keep
  artifact when input confidentiality is required.

## 6) Polish and OOD local evaluation (`data/evaluation/`, TMR-only)

See [`data/evaluation/README.md`](data/evaluation/README.md) for the local
sample inventory, source notes, commands, and interpretation.

The Polish snapshots below are `tmr`-only outputs (`weights: 0,1,0`):

`tmr_ai_sample_result.json`
- `ai_probability`: 0.9631
- `ensemble.label`: `ai`

`tmr_human_author_result.json`
- `ai_probability`: 0.0740
- `ensemble.label`: `human`

`tmr_human_articles_result.json`
- `ai_probability`: 0.6433
- `ensemble.label`: `ai`

Interpretation:
- these are local checks, not production benchmarks;
- they show a calibration/threshold gap on OOD-like chemistry text (`human_articles` is a false-positive AI label);
- this is intentionally a **negative** control and should not be interpreted as a successful detection.
- negative or counterintuitive cases should be kept visible for tuning decisions.

Broader PL/OOD smoke:
- Source fixture: `data/evaluation/polish_pre2020_technical_papers/`.
- Baseline result: `data/evaluation/polish_pre2020_technical_papers/broad_eval_2026-05-24.json`.
- Calibration: `data/evaluation/polish_pre2020_technical_papers/calibration_pl_ood_2026-05-24.json`.
- Source cache: original PDFs and clean extracted full text under
  `data/evaluation/polish_pre2020_technical_papers/sources/`, with SHA-256
  checksums in `sources/manifest.json`.
- Corpus: 12 pre-2020 technical/scientific papers; 11 clean Polish entries used
  for fitting, one mixed English/Polish entry kept for audit only.
- Default ensemble (`0.34,0.33,0.33`) on long 700-word human excerpts:
  `2/11` false positives at threshold `0.5`, `6/11` at threshold `0.4`.
- Source-group split hybrid profile `0.75,0.00,0.25` plus
  `heuristic_weight=0.60`, threshold `0.434552`: heldout window smoke
  `0/12` false positives, `0/3` false negatives; all window records `0/33`
  false positives, `0/8` false negatives, margin `0.095468`.

## 7) Limits and caveats

- PL language support is limited; model quality can drop for Polish text.
- OOD (out-of-distribution) inputs are not guaranteed and can produce unstable
  scores. Thresholds should be adjusted only after local validation.
- Zero-weight models are not loaded to avoid unnecessary compute.
- The heuristic CLI can false-positive formal Polish prose with common AI-like
  markers and false-negative edited AI text without those markers.
- Heuristic scores are useful for fast triage, not enforcement or final labeling.

## 8) Pre-merge checks

Quick checks:
```bash
ai-detector --help
ai-detector-deploy --help
ai-detector-deploy --list-models
ai-detector-heuristic --help
ai-detector-calibrate --help
ai-detector --text "quick smoke test with enough words for heuristic profile metadata" --json
printf "quick smoke test with enough words for heuristic profile metadata\n" > /tmp/ai_detector_input.txt
ai-detector --text-file /tmp/ai_detector_input.txt --json
printf "quick smoke test with enough words for heuristic profile metadata\n" | ai-detector --json
ai-detector-heuristic --text "Warto zauważyć, że to szybki test lokalny z pełnym zdaniem operatora." --json
ai-detector-calibrate \
  --baseline-result data/evaluation/polish_pre2020_technical_papers/broad_eval_windows_2026-05-24.json \
  --output /tmp/ai_detector_calibration.json \
  --id pl_technical_ood_2026_05_24 \
  --grid-step 0.05 \
  --fixed-weights 0.75,0,0.25 \
  --fit-heuristic-weight \
  --heuristic-grid-step 0.05 \
  --max-heuristic-weight 0.60 \
  --split-seed pl-technical-ood-v2
```

Heavy smoke test:
```bash
printf "This is a longer validation snippet for smoke testing the ensemble path.\n" > /tmp/ai_detector_smoke.txt
ai-detector \
  --text-file /tmp/ai_detector_smoke.txt \
  --weights 0.34,0.33,0.33 \
  --max-chunks 4 \
  --batch-size 4 \
  --overlap 64 \
  --quiet \
  --json > /tmp/ai_detector_smoke.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/ai_detector_smoke.json", "r", encoding="utf-8"))

for key in ("experts", "ensemble", "calibration"):
    assert key in result, f"Missing section: {key}"
for expert in ("meld", "tmr", "raid"):
    assert expert in result["experts"], f"Missing expert: {expert}"

print("ok")
PY
```

Expected smoke test outcome:
- command exits 0
- JSON includes `experts.*`, `ensemble`, `calibration`, `weights`, `device`
- `ensemble.label` is `ai` or `human`

Note: full smoke test requires runtime dependencies (including `torch`), so it is
recommended to run these commands after activating the project virtualenv. From a
checkout without package entrypoints, replace `ai-detector` with
`python3 run_ensemble.py`.

## 9) GitHub CI/CD

- `.github/workflows/ci.yml` runs on pull requests and pushes to `main`.
  It checks entrypoint help, syntax, evaluation JSON artifacts, calibration
  regeneration, `pytest`, package build, and `twine check`.
- `.github/workflows/release.yml` builds source/wheel distributions on manual
  dispatch and on GitHub Release publish, then runs `twine check`. Release
  publish also attempts PyPI trusted publishing through the `pypi` environment.
- `.github/workflows/runtime-smoke.yml` is scheduled/manual and downloads cached
  Hugging Face model artifacts before running a real calibrated inference smoke.

## 10) Preloaded daemon (`ai-detector-daemon`)

Use the daemon when you send repeated local scoring requests. It keeps models
preloaded and scores JSONL requests sequentially in one process, so you avoid
CLI cold-start/model-load overhead.

Recommended launch on this CPU (P-core pinning):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-7 ai-detector-daemon --local-files-only --threads 4 --device cpu --quiet
```

- `--device cpu` is explicit but optional because the daemon default is CPU.
- Use `--device auto` only when doing direct accelerator-vs-CPU comparisons.
- `taskset -c 0-7` is used to pin to this host’s P-core cluster in the local
  daemon launch profile.

Protocol: one JSON object per line on stdin, one JSON object per line on stdout.

Choose executable:

```bash
if command -v ai-detector-daemon >/dev/null 2>&1; then
  DAEMON_CMD=(ai-detector-daemon)
else
  DAEMON_CMD=(python3 detector_daemon.py)
fi
```

Lifecycle (single process):
```bash
coproc DAEMON {
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-7 "${DAEMON_CMD[@]}" \
    --local-files-only --threads 4 --device cpu --quiet
}

# Health check
printf '%s\n' '{"command":"health"}' >&"${DAEMON[1]}"
read -r health_resp <&"${DAEMON[0]}"
echo "health: $health_resp"

# Scoring request (uses daemon default profile)
printf '%s\n' '{"text":"This is a review sentence with enough context to score.","threshold":0.5}' >&"${DAEMON[1]}"
read -r score_resp <&"${DAEMON[0]}"
echo "score: $score_resp"

# Shutdown/unload path: request process exit and wait for it to terminate
printf '%s\n' '{"command":"shutdown"}' >&"${DAEMON[1]}"
read -r shutdown_resp <&"${DAEMON[0]}"
echo "shutdown: $shutdown_resp"
wait "$DAEMON_PID"
```

Expected response shapes:
- Health:
  `{"status":"ok","command":"health","loaded_experts":["meld","tmr","raid"],"device":"cpu","threads":4,"local_files_only":true}`
- Scoring (default PL/OOD profile):
  `{"text_preview":"...","weights":{"meld":0.30,"tmr":0.00,"raid":0.10,"heuristic":0.60},"experts":{...},"ensemble":{...},"calibration":{...},"device":"cpu"}`
- Shutdown:
  `{"status":"ok","command":"shutdown","ack":true,"loaded_experts":[]}`

- Sending `{"command":"shutdown"}` is the unload path for this daemon. The daemon is
  designed to be torn down cleanly this way.
- For reliable RSS recovery in operators, process exit/restart is the robust unload
  path; partial Python GC (`gc.collect()`) may not return all memory to OS.

`--experts` subset behavior:
- Start with `--experts tmr,raid` to preload only those two.
- If a request does not provide `weights`, defaults are remapped only to the
  preloaded set. For legacy behavior, launch with `--profile legacy-default`;
  with raw default `0.34,0.33,0.33`, `tmr,raid` becomes `tmr=0.5`, `raid=0.5`
  internally.
- If a request provides explicit positive weight for a non-preloaded expert, the
  daemon returns an error.

Output behavior and privacy:
- JSON output shape matches the CLI contract used by `ai-detector`, including
  `experts`, `ensemble`, and `calibration`.
- `ai_probability` and `human_probability` are raw uncalibrated values unless a
  calibrated scoring path is configured.
- `text_preview` is still emitted exactly like the CLI and keeps the first
  250 characters; avoid sending sensitive text if this output may be logged.

## 11) Runtime dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors
```

## 12) Personal Polish style similarity (`personal_style_pl`)

A separate, local tool that learns **your** Polish writing style from your own
samples and scores how much new text resembles it. It reuses this repo's
Polish tokenizer and AI-marker phrase lists, and can attach the existing
heuristic AI-likeness score via `--with-heuristics`.

**What it does:** builds a personal style profile, scores/ranks drafts by
similarity to your style, explains which features match/diverge, and suggests
conservative, meaning-preserving edits.

**What it does NOT do:** it is **style similarity, not authorship verification**
and not proof of AI authorship. It never invents facts and never rewrites
names, dates, numbers, quotes, URLs, citations, or code.

### Setup (single Python 3.12 env)

StyloMetrix pins `spacy==3.7.2`, which has no wheels above CPython 3.12, so the
whole project standardizes on **Python 3.12** with `numpy<2`. Use the scripted,
reproducible setup (requires [`uv`](https://astral.sh/uv)):

```bash
./scripts/setup_style_env.sh
```

This builds `.venv` on 3.12, installs `.[test,style,style-stylometrix]`, and
installs the Polish `pl_nask` model `--no-deps` from the TLS-valid Hugging Face
mirror (the official IPI PAN host has an expired certificate). Surface-only mode
needs just `pip install -e ".[style]"` and runs with `--no-stylometrix`.

### Commands

```bash
# Build a profile from a folder of your texts (surface-only / degraded mode)
python -m personal_style_pl.cli build-profile \
  --samples-dir examples/my_style_samples \
  --output artifacts/profile.joblib --no-stylometrix

# Build with StyloMetrix (172 Polish features; needs the style-stylometrix env)
python -m personal_style_pl.cli build-profile \
  --samples-dir examples/my_style_samples --output artifacts/profile.joblib

# Build from CSV (columns: text,source,date,genre)
python -m personal_style_pl.cli build-profile --csv examples/style_samples.csv \
  --text-col text --output artifacts/profile.joblib

# Score one text (JSON), optionally attaching heuristic AI-likeness
python -m personal_style_pl.cli score --profile artifacts/profile.joblib \
  --text-file examples/candidates/draft_a.txt --json --with-heuristics

# Rank candidate drafts -> CSV
python -m personal_style_pl.cli rank --profile artifacts/profile.joblib \
  --candidates-dir examples/candidates --output artifacts/ranking.csv

# Human-readable profile summary
python -m personal_style_pl.cli describe-profile \
  --profile artifacts/profile.joblib --output artifacts/profile_summary.md

# Conservative suggestions / deterministic edit
python -m personal_style_pl.cli suggest-edits --profile artifacts/profile.joblib \
  --text-file examples/candidates/draft_a.txt --output artifacts/suggestions.md --mode light
python -m personal_style_pl.cli edit --profile artifacts/profile.joblib \
  --text-file examples/candidates/draft_a.txt --output artifacts/draft_a.edited.txt --mode light

# Optional supervised mine-vs-other classifier (CSV labels: mine|other)
python -m personal_style_pl.cli train-supervised --csv data/contrast.csv \
  --text-col text --label-col label --output artifacts/supervised_style_model.joblib
```

### Score interpretation

`style_match_score` is 0–100. Labels: **80–100** `close_to_my_style`,
**55–79** `mixed`, **0–54** `far_from_my_style`; `insufficient_text` when the
candidate is too short. `confidence` is `high`/`medium`/`low` based on profile
size and candidate length. The score blends a robust per-feature z-distance
(70%) with cosine-to-centroid (30%), with temperature calibrated from your
training chunks.

### Limitations & ethics

- Provide **at least 10–20 samples or 5,000+ words** for a meaningful profile;
  smaller profiles emit a weak-profile warning and force low confidence.
- N-grams are topic-sensitive and off by default (`--include-ngrams`).
- Domain/genre mismatch reduces reliability.
- Use it to understand and match *your own* style — not to impersonate others or
  to evade AI detectors.

### Rich metrics & AI-leaning markers (v2)

An interpretable, **length-normalized** metric layer plus an **AI-direction
overlay**. It measures length-robust lexical diversity (MATTR / MTLD / HD-D,
borrowed from [LexicalRichness](https://github.com/LSYS/LexicalRichness)),
sentence-length **burstiness**, repeated higher-order n-grams, and per-1,000-token
densities of em-dashes, boilerplate, transition and hedge phrases. These features
are **on by default** in every profile (disable with `--no-rich`). An optional,
gated [TextDescriptives](https://github.com/HLasse/TextDescriptives) block adds
dependency-distance and POS-proportion features over `pl_nask` (entropy/perplexity
are excluded — they are NaN for Polish).

The `ai-markers` command reads those metrics against literature-backed AI
directionality (e.g. low burstiness, em-dash and boilerplate over-use, repeated
n-grams) and, when given a profile, scores each marker **against your own
baseline**:

```bash
# Interpretable AI-leaning report vs your style (advisory; abstains on Polish)
python -m personal_style_pl.cli ai-markers \
  --text-file examples/candidates/draft_b.txt \
  --profile artifacts/profile.joblib --json

# Optional Polish-LM perplexity signal (papuGaPT2; ~500 MB, fetched by the setup
# script). Build the baseline with it, then score the perplexity markers too:
python -m personal_style_pl.cli build-profile --samples-dir examples/my_style_samples \
  --output artifacts/profile_ppl.joblib --no-stylometrix --with-perplexity
python -m personal_style_pl.cli ai-markers --text-file examples/candidates/draft_b.txt \
  --profile artifacts/profile_ppl.joblib --with-perplexity --json

# The lean detector can attach the same block (lazy-imports; no effect on its deps):
python heuristic_detector.py --text-file examples/candidates/draft_b.txt --json --rich
```

`suggest-edits` also appends an advisory "AI-leaning markers" section, and
`train-supervised --features rich` trains the mine-vs-other classifier on
surface + rich features.

**Stance (important):** these markers are an interpretable **heuristic, not proof
of authorship**. On **Polish / out-of-distribution** input the overlay **abstains**
(`ood_or_unreliable: true`, low confidence) rather than emit a confident AI/human
label — false positives are the dominant harm, and Polish-LM perplexity did *not*
cleanly separate human from AI-assisted text in our audit, so it is advisory only.

**Polish-register calibration (v2.1).** A windowed human-vs-AI study on Polish
polymer-chemistry prose (pre-2020 human papers vs generated AI text) found that some
markers are unreliable on this register and are therefore **advisory-only on Polish**
(shown but not counted in the score): `mattr` (terse, term-repeating scientific prose
has *lower* diversity than verbose AI — the polarity flips), `em_dash_per_1k` (Polish
journals use en-dashes for numeric ranges, e.g. `23–70°C`), and `repeated_4gram_ratio`
(real papers repeat terminology). The markers that separated correctly — and are
**counted** on Polish — are `burstiness_coeff`, `sentence_len_cv`, `transition_per_1k`,
and `boilerplate_per_1k`. Median perplexity is surfaced as an **advisory
`perplexity_flag`** (calibrated threshold ≈47; modest TPR≈0.65 / FPR≈0.22), never a
verdict. On non-Polish input all markers are counted, since e.g. em-dash overuse is a
genuine English tell. Each marker row carries `reliable`/`counted` so you can see what
fed the score. (Caveat: calibrated on a small, single-author/single-AI-model corpus —
treat as a domain heuristic, not a universal classifier; terse "humanized" AI without
boilerplate/transitions can still evade the marker-based score.)

### Judging AI-generated vs human text (recipe + honest limits)

To use this repo to **discriminate AI-generated from human text** (rather than just style
similarity), treat it as an **advisory triage**, not a verdict:

```bash
# 1. Build a KNOWN-HUMAN baseline from reference text you trust is human
#    (e.g. the author's pre-LLM / pre-2020 writing). Rich metrics are on by default.
python -m personal_style_pl.cli build-profile --samples-dir known_human_samples/ \
  --output artifacts/human_baseline.joblib --no-stylometrix --with-perplexity

# 2. Score the candidate text against that human baseline
python -m personal_style_pl.cli ai-markers --text-file candidate.txt \
  --profile artifacts/human_baseline.joblib --with-perplexity --json
```

**How to read the output:**
- `ai_leaning_score` (0–100) = % of **counted** markers leaning AI vs the human baseline. Higher =
  more AI-leaning. It is **relative to the baseline**, not an absolute probability.
- Each marker row has `leaning` (`AI-leaning` / `matches` / `more-human-than-you` /
  `advisory_only`) and `counted` (whether it fed the score). Trust the **counted** ones.
- `perplexity_flag` (with `--with-perplexity`): `leans_AI_low_perplexity` vs
  `within_or_above_human_range` — advisory, calibrated on Polish (threshold ≈47).
- `ood_or_unreliable: true` + low `confidence` on Polish → **abstain**: report as advisory, never
  as a definitive AI/human label.

**What it can and cannot do (validated on Polish polymer-chemistry prose, v2.1):**
- ✅ Reliably separates **naive AI** (formulaic phrasing, signposted transitions, uniform rhythm,
  low perplexity) from human: in a held-out study the real human text scored 0 while AI averaged ~29.
- ⚠️ **Misses terse "humanized" AI** that drops boilerplate/transitions — the marker score can read
  0; only the (weak) perplexity signal hints. This is a hard limit of interpretable markers.
- ⚠️ **Polish is out-of-distribution** — the overlay abstains and only some markers count (see the
  v2.1 calibration note above). Do not emit confident AI/human verdicts for Polish.
- ⚠️ Use clean prose: PDF/LaTeX-extraction noise inflates perplexity and repetition metrics.

For a stronger, supervised approach when you have labelled examples, see
`train-supervised --features rich` (a mine-vs-other / human-vs-AI classifier on surface+rich
features). The discrimination methodology used to calibrate v2.1 (windowed human-vs-AI study) is
summarised in the `ai_markers.py` module docstring.

### License note

`stylo_metrix` and the `pl_nask` model are **GPL-3.0**. They are optional extras;
if you redistribute the project as a whole with them, GPL obligations apply.
The rich-metrics extras are permissive: **LexicalRichness** (MIT) and
**TextDescriptives** (Apache-2.0); the optional **papuGaPT2** model
(`dkleczek/papuGaPT2`) is MIT.
