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
- `--revision` defaults to `main`; treat `main` as a mutable pointer.
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

Privacy note:
- `text_preview` in JSON output is the first 250 characters of raw input text.
- This preview is emitted as part of CLI output and can be copied into logs by operators;
  avoid sending sensitive or regulated text through the same output channel unless
  log redaction is in place.
- Keep `text_preview` as a debug aid only; it is not a sanitized/safe-to-keep
  artifact when input confidentiality is required.

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
- `ai_probability`: 0.6433
- `ensemble.label`: `ai`

Interpretation:
- these are local checks, not production benchmarks;
- they show a calibration/threshold gap on OOD-like chemistry text (`human_articles` is a false-positive AI label);
- this is intentionally a **negative** control and should not be interpreted as a successful detection.
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

## 8) Preloaded daemon (`ai-detector-daemon`)

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

# Scoring request (uses daemon default weights)
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
- Scoring (default request weights):
  `{"text_preview":"...","weights":{"meld":0.34,"tmr":0.33,"raid":0.33},"experts":{...},"ensemble":{...},"calibration":{...},"device":"cpu"}`
- Shutdown:
  `{"status":"ok","command":"shutdown"}`

- Sending `{"command":"shutdown"}` is the unload path for this daemon. The daemon is
  designed to be torn down cleanly this way.
- For reliable RSS recovery in operators, process exit/restart is the robust unload
  path; partial Python GC (`gc.collect()`) may not return all memory to OS.

`--experts` subset behavior:
- Start with `--experts tmr,raid` to preload only those two.
- If a request does not provide `weights`, defaults are remapped only to the
  preloaded set (for example, with default `0.34,0.33,0.33`, `tmr,raid` becomes
  `tmr=0.5`, `raid=0.5` internally).
- If a request provides explicit positive weight for a non-preloaded expert, the
  daemon returns an error.

Output behavior and privacy:
- JSON output shape matches the CLI contract used by `ai-detector`, including
  `experts`, `ensemble`, and `calibration`.
- `ai_probability` and `human_probability` are raw uncalibrated values unless a
  calibrated scoring path is configured.
- `text_preview` is still emitted exactly like the CLI and keeps the first
  250 characters; avoid sending sensitive text if this output may be logged.

## 9) Runtime dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors
```
