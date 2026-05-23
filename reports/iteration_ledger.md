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

## Current Evaluation Evidence

The tracked Polish TMR-only fixtures are local smoke checks, not calibrated
benchmarks. The chemistry-article excerpt is deliberately kept as a negative
OOD result: current TMR-only scoring labels it `ai` with
`ai_probability = 0.6433479888364673`, despite the source text being human
pre-2020 Polish chemistry/polymer prose.
