# Polish Evaluation Notes

This directory contains small local fixtures for smoke-checking Polish text
classification behavior. These files are not a calibrated benchmark.

## Files

- `ai_polish_generated_style_sample.txt`: synthetic Polish chemistry/polymer
  prose written to look AI-generated.
- `human_author_style_excerpt.txt`: bounded excerpt from the local thesis
  author-style sample.
- `human_polish_chemistry_articles_excerpt.txt`: bounded excerpt assembled from
  Polish chemistry/polymer articles published before 2020.
- `polish_pre2020_technical_papers/`: broader pre-2020 Polish technical-paper
  source manifest, span-preserved short/long excerpts, and baseline scores.
- `sources.json`: source metadata for the local human samples.
- `tmr_*_result.json`: stored TMR-only smoke outputs.

## TMR-Only Smoke Commands

```bash
.venv/bin/ai-detector \
  --weights 0,1,0 \
  --device cpu \
  --batch-size 4 \
  --max-chunks 8 \
  --overlap 64 \
  --json \
  --text-file data/evaluation/ai_polish_generated_style_sample.txt
```

Repeat with:

```bash
data/evaluation/human_author_style_excerpt.txt
data/evaluation/human_polish_chemistry_articles_excerpt.txt
```

## Current Results

- `tmr_ai_sample_result.json`: `ai_probability = 0.963`, label `ai`.
- `tmr_human_author_result.json`: `ai_probability = 0.074`, label `human`.
- `tmr_human_articles_result.json`: `ai_probability = 0.643`, label `ai`.

The chemistry-article result is a negative OOD/threshold signal, not a success.
Keep it visible as a negative fixture when tuning thresholds and adding calibration.

## Broader Polish Technical Paper Smoke

Fixture:

```text
data/evaluation/polish_pre2020_technical_papers/excerpts.jsonl
data/evaluation/polish_pre2020_technical_papers/windows.jsonl
data/evaluation/polish_pre2020_technical_papers/manifest.json
data/evaluation/polish_pre2020_technical_papers/sources/manifest.json
data/evaluation/polish_pre2020_technical_papers/sources/pdf/
data/evaluation/polish_pre2020_technical_papers/sources/text/
data/evaluation/polish_pre2020_technical_papers/broad_eval_2026-05-24.json
data/evaluation/polish_pre2020_technical_papers/broad_eval_windows_2026-05-24.json
data/evaluation/polish_pre2020_technical_papers/calibration_pl_ood_2026-05-24.json
```

This set contains 12 pre-2020 technical/scientific papers from different
backgrounds. Use the 11 `pl_clean` entries as known-human Polish controls. The
mixed English/Polish maritime article is retained for audit/reference but is not
used for Polish-only fitting.

Original downloaded PDFs and clean extracted full text are stored under
`polish_pre2020_technical_papers/sources/` with SHA-256 checksums in
`sources/manifest.json`.

Baseline run settings:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 \
  .venv/bin/python run_ensemble.py \
  --weights 0.34,0.33,0.33 \
  --max-chunks 4 \
  --batch-size 4 \
  --overlap 64 \
  --device cpu \
  --local-files-only \
  --json
```

Long 700-word clean Polish human excerpts:
- heuristic: `0/11` false positives at threshold `0.5`, mean `0.329`;
- default ensemble `0.34,0.33,0.33`: `2/11` false positives, mean `0.426`;
- MELD: `0/11` false positives, mean `0.316`;
- TMR: `6/11` false positives, mean `0.549`;
- RAID/MAGE: `5/11` false positives, mean `0.415`.

The packaged PL/OOD runtime profile uses model weights
`meld,tmr,raid = 0.75,0.00,0.25`, blends in the heuristic with
`heuristic_weight = 0.60`, and uses threshold `0.434552`. The expanded window
smoke uses three non-overlapping 220-word windows per clean Polish paper and a
source-group split. It gave heldout `0/12` false positives and `0/3` false
negatives; across all window records it gave `0/33` false positives, `0/8`
false negatives, and margin `0.095468`. Treat this as a local operating-point
baseline, not probability calibration.

Regenerate the calibration file with:

```bash
python3 calibrate_detector.py \
  --baseline-result data/evaluation/polish_pre2020_technical_papers/broad_eval_windows_2026-05-24.json \
  --output data/evaluation/polish_pre2020_technical_papers/calibration_pl_ood_2026-05-24.json \
  --id pl_technical_ood_2026_05_24 \
  --grid-step 0.05 \
  --fixed-weights 0.75,0,0.25 \
  --fit-heuristic-weight \
  --heuristic-grid-step 0.05 \
  --max-heuristic-weight 0.60 \
  --split-seed pl-technical-ood-v2 \
  --view all \
  --human-quality pl_clean
```

Use it at runtime with:

```bash
.venv/bin/ai-detector \
  --profile pl-technical-ood \
  --text-file ./input.txt \
  --json
```

The JSON calibration file is the reproducible source for the packaged
`pl-technical-ood` runtime profile. It is an operating-point calibration, not
probability calibration.

## Fast Heuristic Smoke

Use the heuristic CLI for cheap marker-based triage before loading model weights:

```bash
.venv/bin/ai-detector-heuristic \
  --json \
  --text-file data/evaluation/ai_polish_generated_style_sample.txt
```

Repeat with the same local text files listed above. Heuristic scores are raw,
uncalibrated rule signals and should be compared against TMR/ensemble outputs,
not used as final labels.

## Heuristic vs Ensemble Smoke

Earlier six-sample comparison used the three files above plus 700-word cleaned
excerpts from:

- `/home/jakub/projects/doktorat/frontmatter/abstract_pl.tex`
- `/home/jakub/projects/doktorat/chapters/01_cel_pracy.tex`
- `/home/jakub/projects/doktorat/chapters/02_wstep.tex`

Ensemble config: `weights=0.34,0.33,0.33`, `max_chunks=4`,
`batch_size=4`, `overlap=64`, `device=cpu`.

Result on six samples:
- Pearson heuristic-vs-ensemble correlation: `0.690`
- Spearman heuristic-vs-ensemble correlation: `0.771`

Interpretation: this is a small smoke check showing a moderate positive
relationship, not calibration. Keep using the heuristic for triage and the
ensemble for detailed model scoring.
