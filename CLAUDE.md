# CLAUDE.md — repo guide for agents

This repo has **two subsystems** that share a single **Python 3.12** virtualenv (`.venv`):

1. **AI-text detector** (`run_ensemble.py`, `heuristic_detector.py`, `deploy_meld.py`,
   `detector_daemon.py`, `calibrate_detector.py`) — decides whether text is AI-generated
   (Polish/English). ModernBERT ensemble (MELD/TMR/RAID) + a fast stdlib heuristic scorer,
   with operating-point calibration.
2. **`personal_style_pl/`** — Polish *personal writing-style similarity*: learn the user's
   own style from their samples, then score/rank/explain/edit new text by resemblance.
   This is **style similarity, not authorship/AI-authorship proof**. It reuses the
   detector's tokenizer + phrase lists and can attach AI-likeness via `--with-heuristics`.

## Environment (read first)

- One Python 3.12 `.venv`. StyloMetrix pins `spacy==3.7.2` (no wheels >cp312), so the whole
  project is pinned to 3.12 with `numpy<2`. Rebuild with `./scripts/setup_style_env.sh`
  (needs `uv`). Run everything via `.venv/bin/python` or `source .venv/bin/activate`.
- Detector model weights live in `meld_model/`, `tmr_model/`, `raid_model/` (git-ignored
  artifacts; deploy with `ai-detector-deploy --all`). Do not edit them unless asked.

## Common commands

```bash
# Detect AI text
.venv/bin/python -m run_ensemble --text-file t.txt --profile pl-technical-ood --json
.venv/bin/python heuristic_detector.py --text "..." --json

# Personal style similarity (build once, then score/rank/explain/edit)
.venv/bin/python -m personal_style_pl.cli build-profile --samples-dir DIR --output artifacts/p.joblib
.venv/bin/python -m personal_style_pl.cli score --profile artifacts/p.joblib --text-file d.txt --json --with-heuristics
.venv/bin/python -m personal_style_pl.cli rank|describe-profile|suggest-edits|edit|train-supervised --help
```

## Tests

```bash
.venv/bin/python -m pytest -q                          # full suite (detector + style)
.venv/bin/python -m pytest -q --ignore=tests/personal_style   # detector only (lean CI parity)
```

## Where to look

- Usage details: `README.md` (§1–11 detector, §12 `personal_style_pl`).
- Rules: `AGENTS.md` (detector/ops) and `personal_style_pl/AGENTS.md` (style feature).
- Design/spec/plan: `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Every CLI supports `--help` (authoritative flags).
