# AI Detector CLI

This repo provides a 3-expert ensemble CLI for text classification:
MELD, TMR, and MAGE (ModernBERT from `raid_model`).

## CLI install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Then use the installed entrypoints:

```bash
ai-detector --help
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

## 2) Model folders and deployment

Use the installed deploy entrypoint to pull Hugging Face models into local
folders.

```bash
ai-detector-deploy --model-id anon-review-meld-2026/meld --target-dir ./meld_model
ai-detector-deploy --model-id Oxidane/tmr-ai-text-detector --target-dir ./tmr_model
ai-detector-deploy --model-id GeorgeDrayson/modernbert-ai-detection-raid-mage --target-dir ./raid_model
```

Defaults and behavior:
- `ai-detector-deploy` default target is `./models/meld` (override with `--target-dir`).
- Re-running the same command is idempotent; unchanged files are skipped.
- A manifest is written as `ai_detector_model_manifest.json` in each target folder.

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
# default weights 34/33/33, default threshold 0.5
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
```

Notes:
- `--weights` accepts two or three values: `meld,tmr,(raid)`.
- `--batch-size` controls chunk batch size for scoring.
- `--max-chunks` caps the number of chunks scored per expert.
- `--threshold` compares only the final ensemble value.
- `--text`, `--text-file`, stdin pipe, and `--json` output mode are supported.
- `--quiet` suppresses third-party stderr chatter during model loading/scoring;
  user-facing `Error: ...` messages are still printed on failure.

## 4) Output contract (AI / human)

JSON fields used by operators:
- `weights`
- `experts.meld`, `experts.tmr`, `experts.raid` each with:
  - `ai_score`, `human_score`, `ai_probability`, `human_probability` when loaded
  - `loaded`, `chunks`, optional `notes`
- `ensemble.ai_probability`, `ensemble.human_probability`, `ensemble.threshold`, `ensemble.label`
- `calibration.status`, `calibration.calibrated`, `calibration.message`

Decision logic:
- `ensemble.ai_probability` is the weighted average of loaded experts.
- `ensemble.label` is `ai` if `ensemble.ai_probability >= threshold`, else `human`.
- `ensemble.human_probability` is `1 - ensemble.ai_probability`.

Scoring details:
- For loaded experts, `human_probability = 1 - ai_probability`.
- For skipped experts (`loaded: false`, typically due to zero weight), all score/probability
  fields are `null` and `notes` explains the skip.
- Without calibration configuration, all probabilities are **raw uncalibrated scores**.

## 5) Polish and OOD local evaluation (`data/evaluation/`, TMR-only)

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
- `ai_probability`: 0.5218
- `ensemble.label`: `ai`

Interpretation:
- these are local checks, not production benchmarks;
- they show a calibration/threshold gap on OOD-like chemistry text (`human_articles` near boundary with AI label);
- negative or counterintuitive cases should be kept visible for tuning decisions.

## 6) Limits and caveats

- PL language support is limited; model quality can drop for Polish text.
- OOD (out-of-distribution) inputs are not guaranteed and can produce unstable
  scores. Thresholds should be adjusted only after local validation.
- Zero-weight models are not loaded to avoid unnecessary compute.

## 7) Pre-merge checks

Quick checks:
```bash
ai-detector --help
ai-detector --text "quick smoke test" --json
printf "quick smoke test\n" > /tmp/ai_detector_input.txt
ai-detector --text-file /tmp/ai_detector_input.txt --json
printf "quick smoke test\n" | ai-detector --json
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

## 8) Runtime dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors
```
