# Iteration Ledger

This ledger records the tracked hardening iterations for the AI detector CLI.
It is intentionally concise: detailed reviewer comments live in the agent
thread, while this file gives future reviewers repo-local evidence that the
iteration/review loop happened.

## Commit Iterations

| # | Commit | Scope | Review outcome |
|---|---|---|---|
| 1 | `f81bc78` | Initial baseline import | Baseline for initial review. |
| 2 | `98d879a` | Harden ensemble CLI, deployment, tests, and evaluation fixtures | Built from initial GPT-5.5 review findings. |
| 3 | `3d69fcc` | Package CLI entrypoints and evaluation docs | Reviewed; docs gaps carried into later fixes. |
| 4 | `7269af4` | Harden inference CLI error reporting | Reviewed; model loader filesystem gap fixed next. |
| 5 | `58d2827` | Wrap model loader filesystem errors | Reviewed; typing issue fixed later. |
| 6 | `d3dedab` | Document installed CLI workflow | Reviewed; docs scope follow-up added. |
| 7 | `2e2f006` | Clarify expert loader typing | Reviewed clean. |
| 8 | `2d1dbc4` | Align documentation scope with evaluation notes | Follow-up to documentation review. |
| 9 | `b307a46` | Add quiet inference mode | Reviewed; fd-level stderr gap fixed later. |
| 10 | `f7ddb2a` | Add packaging CLI contract tests | Reviewed; packaging-test gaps fixed later. |
| 11 | `fe1b2f7` | Harden deploy CLI errors | Reviewed; subprocess/success coverage added later. |
| 12 | `b9efae2` | Suppress fd-level quiet stderr | Reviewed clean. |
| 13 | `b428154` | Tighten packaging contract tests | Reviewed clean. |
| 14 | `a4dd5e3` | Cover deploy failure subprocess behavior | Reviewed; subprocess test tightened later. |
| 15 | `ada6bda` | Cover deploy success path | Reviewed; call-order and argv wiring fixed later. |
| 16 | `6e41c88` | Tighten deploy subprocess failure test | Follow-up to deploy subprocess review. |
| 17 | `67e28a8` | Cover JSON CLI contract in subprocess | Reviewed; nested contract assertions fixed later. |
| 18 | `dc6c125` | Verify deploy success call order | Follow-up to deploy success-path review. |
| 19 | `35d342f` | Assert nested JSON CLI contract | Reviewed in final audit; OOD evidence and ledger gaps fixed next. |
| 20 | `HEAD` | Refresh Polish OOD evidence and add process ledger | Final closeout iteration, to be reviewed after commit. |
| 21 | `HEAD` | Add fast heuristic detector CLI | Local browser-detector-style heuristic port with parity smoke tests. |
| 22 | `HEAD` | Compare heuristic with ensemble samples | Six-sample heuristic-vs-ensemble smoke showed moderate positive correlation. |
| 23 | `HEAD` | Add broad pre-2020 Polish technical-paper smoke | Source fixture and baseline scores show TMR/RAID over-call PL/OOD human text; provisional calibration added. |
| 24 | `HEAD` | Add PL/OOD operating-point calibration | Runtime profile and `--calibration-file` support; source-group split model calibration set weights `0.75,0.00,0.25`, threshold `0.513591`. |
| 25 | `HEAD` | Add expanded window smoke and hybrid PL/OOD profile | Three non-overlapping windows per paper plus heuristic blend reduced observed FP/FN on expanded local smoke. |
| 26 | `HEAD` | Make hybrid profile reproducible and persist source papers | Calibration artifact now regenerates the shipped hybrid profile from expanded windows; original PDFs/full clean text are cached with checksums. |

## Current Evaluation Evidence

The tracked Polish TMR-only fixtures are local smoke checks, not calibrated
benchmarks. The chemistry-article excerpt is deliberately kept as a negative
OOD result: current TMR-only scoring labels it `ai` with
`ai_probability = 0.6433479888364673`, despite the source text being human
pre-2020 Polish chemistry/polymer prose.

The broad pre-2020 Polish technical-paper fixture is stored under
`data/evaluation/polish_pre2020_technical_papers/`. The 2026-05-24 baseline
keeps 11 clean Polish known-human entries and one mixed-language audit entry.
Default legacy weights `0.34,0.33,0.33` produced `2/11` false positives on long
clean Polish human excerpts at threshold `0.5`. The expanded window smoke uses
three non-overlapping 220-word windows per clean Polish paper. The packaged
`pl-technical-ood` profile uses model weights `0.75,0.00,0.25`, heuristic
weight `0.60`, and threshold `0.434552`; on the expanded window smoke it
produced heldout `0/12` false positives and `0/3` false negatives, and all-window
`0/33` false positives and `0/8` false negatives. The runtime calibration is
regenerated from `broad_eval_windows_2026-05-24.json` with fixed model weights
and fitted heuristic weight. Original source PDFs and clean extracted full text
are stored under `data/evaluation/polish_pre2020_technical_papers/sources/`
with SHA-256 checksums.
