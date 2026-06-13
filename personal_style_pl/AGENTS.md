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

## Environment
- Single Python 3.12 venv (`.venv`). Surface features + scoring need only
  numpy/pandas/scikit-learn/scipy/joblib; StyloMetrix is optional and failable
  (`--no-stylometrix` runs in degraded surface-only mode).
