# AGENTS.md — personal_style_pl

## Project rules
- Preserve factual content in editing mode (`edit` / `suggest-edits`).
- Prefer transparent scoring and explanations over opaque claims.
- Add tests for every core function.
- Run pytest after code changes: `.venv/bin/python -m pytest tests/personal_style -q`.
- Keep the CLI examples in `README.md` working.
- Avoid hidden network calls or runtime model downloads. StyloMetrix/`pl_nask`
  setup is explicit and scripted (`scripts/setup_style_env.sh`).
- No external LLM API calls in the default implementation.
- This is **style similarity**, not authorship verification or proof of AI authorship.
- Rich metrics (`features/rich_metrics.py`) are interpretable and length-normalized
  (per-1,000-token densities, MATTR/MTLD, burstiness) and are on by default in the
  profile baseline (`--no-rich` to disable). TextDescriptives and papuGaPT2
  perplexity (`features/perplexity_features.py`, opt-in `--with-perplexity`) are
  **gated**: lazy-imported, model-gated tests, degrade gracefully if absent.
- The `ai-markers` overlay is advisory: **never emit a confident AI/human label for
  Polish/OOD** — abstain (`ood_or_unreliable`, low confidence). Perplexity is an
  advisory signal, not a verdict.

## Environment
- Single Python 3.12 venv (`.venv`). Surface features + scoring need only
  numpy/pandas/scikit-learn/scipy/joblib; StyloMetrix is optional and failable
  (`--no-stylometrix` runs in degraded surface-only mode).
