# AGENTS.md (ai_detector documentation scope)

## Scope

Allowed file set for this workspace role:
- `README.md`
- `AGENTS.md`

Do not edit source code, tests, or model artifacts in this role.

## Documentation target

Keep docs short, operator-focused, and runnable.
Document the CLI in a way that another agent can:
- deploy models
- run inference
- interpret outputs
- validate quickly
- run smoke checks

## Mandatory content checklist

`README.md` MUST include all of the points below:
- Ensemble review summary for MELD / TMR / MAGE
- Explicit local model folders (`meld_model`, `tmr_model`, `raid_model`)
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
- quick checks from README
- heavy smoke test from README
- output keys expected by operator scripts are still present (`experts.*`, `ensemble`, `calibration`)

If anything changes in CLI behavior, update this checklist immediately.
If the output contract changes, update `README.md` and relevant tests in the same turn
and keep negative evaluation outcomes visible (no hiding of poor OOD/PL cases).

## Development governance

- Coding and code modifications are done by dedicated small Spark workers, not in this role.
- This role is reviewable and self-contained: edit only these two files and keep the contract stable.
