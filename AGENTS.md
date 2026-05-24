# AGENTS.md (ai_detector production scope)

## Scope

This repo now includes detector runtime code, calibration fixtures, tests, and
GitHub workflows. Agents may edit source, tests, docs, evaluation metadata, and
workflow files when the task requires it.

Do not edit model artifact directories (`meld_model`, `tmr_model`, `raid_model`)
unless the user explicitly asks for model deployment or refresh.

## Documentation target

Keep docs short, operator-focused, and runnable.
Document the CLI in a way that another agent can:
- deploy models
- run inference
- interpret outputs
- validate quickly
- run smoke checks
- use calibration files

## Mandatory content checklist

`README.md` MUST include all of the points below:
- Ensemble review summary for MELD / TMR / MAGE
- Explicit local model folders (`meld_model`, `tmr_model`, `raid_model`)
- Hugging Face model sources for each expert and the one-command
  `ai-detector-deploy --all` deployment path
- Input/Output review:
  - AI / human decision logic
  - `ai_probability` and `human_probability` meaning
  - explicit note that these are raw uncalibrated scores when no calibration is configured
- Inference command forms:
  - `--text`
  - `--text-file`
  - stdin (pipe)
  - JSON output mode
- Quick tests and one heavy smoke test
- Limitations for Polish (PL) and OOD inputs

## Review rule after each iteration

After any doc change, run and verify:
- `ai-detector --help` after package install, or `python3 run_ensemble.py --help`
  from a checkout fallback
- `ai-detector-deploy --help` and `ai-detector-deploy --list-models` after
  package install, or `python3 deploy_meld.py --help` and
  `python3 deploy_meld.py --list-models` from a checkout fallback
- `ai-detector-heuristic --help` after package install, or
  `python3 heuristic_detector.py --help` from a checkout fallback
- `ai-detector-calibrate --help` after package install, or
  `python3 calibrate_detector.py --help` from a checkout fallback
- quick checks from README
- heavy smoke test from README
- output keys expected by operator scripts are still present (`experts.*`, `ensemble`, `calibration`)

If anything changes in CLI behavior, update this checklist immediately.
If the output contract changes, update `README.md` and relevant tests in the same turn
and keep negative evaluation outcomes visible (no hiding of poor OOD/PL cases).

## Development governance

- Keep runtime changes covered by unit tests.
- Treat model weights as external runtime data dependencies, not source files.
  Use `ai-detector-deploy --all` for the packaged Hugging Face model set unless
  the user explicitly asks for a custom model deployment or refresh.
- Keep source-distribution fixtures in sync through `MANIFEST.in`; tests and
  calibration examples rely on `data/evaluation/` being present in the sdist.
- Keep negative PL/OOD outcomes visible; do not hide false-positive cases when
  tuning weights, thresholds, or calibration files.
- Calibration files are operating-point calibrations unless a probability
  calibration method is explicitly implemented and validated.

## Daemon usage guidance

- Prefer `ai-detector-daemon` for repeated local scoring on the same machine
  when avoiding per-command model load dominates runtime.
- Keep runs reproducible: pin thread/environment settings in benchmark and smoke
  commands (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `taskset`, `--threads`,
  `--device`), and record those settings in notes/artifacts.
- Use JSONL for all daemon traffic (one request object per line) and avoid raw
  JSON dumps of sensitive text when possible.
- Always shut down the daemon at the end of usage: send `{"command":"shutdown"}`,
  otherwise terminate the process if control is lost.
