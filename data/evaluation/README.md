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
Keep this visible when tuning thresholds or adding calibration.
