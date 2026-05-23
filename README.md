# AI Detector Deployment

This repository now ships a reproducible deploy flow for three Hugging Face AI
detectors:

- `anon-review-meld-2026/meld`
- `Oxidane/tmr-ai-text-detector`
- `GeorgeDrayson/modernbert-ai-detection-raid-mage`

## Deploy the models

Run each deployment from this working directory:

```bash
python3 deploy_meld.py --target-dir ./meld_model
```

or:

```bash
python3 deploy_meld.py --model-id Oxidane/tmr-ai-text-detector --target-dir ./tmr_model
python3 deploy_meld.py --model-id GeorgeDrayson/modernbert-ai-detection-raid-mage --target-dir ./raid_model
```

Deploying both with explicit directories is recommended. The default target for
`deploy_meld.py` is `./models/meld`, so you can still run:

```bash
python3 deploy_meld.py --model-id anon-review-meld-2026/meld
python3 deploy_meld.py --model-id Oxidane/tmr-ai-text-detector
```

Running again skips files that are already present and unchanged.

## Run ensemble inference

Use `run_ensemble.py` to run both experts and combine their AI scores:

```bash
# 34/33/33 default (MELD/TMR/RAID)
python3 run_ensemble.py --text "The sky is blue and clear."

# custom weights: 70% MELD, 20% TMR, 10% RAID
python3 run_ensemble.py --weights 0.7,0.2,0.1 --text "The sky is blue and clear."

# from a file, JSON output
python3 run_ensemble.py --text-file ./input.txt --json
```

The script prints each expert score and a combined ensemble score:

- `experts.meld.ai_probability`
- `experts.tmr.ai_probability`
- `experts.raid.ai_probability`
- `ensemble.ai_probability`

By default it compares the ensemble score against `--threshold 0.5`.

## Runtime dependencies

Create the environment and install packages if needed:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors
```

Then run the inference commands in the same environment.

## What each deployment folder contains

### `meld_model/`
- `.gitattributes`
- `README.md`
- `config.json`
- `meld_config.json`
- `model.safetensors`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`

### `tmr_model/`
- `.gitattributes`
- `README.md`
- `config.json`
- `merges.txt`
- `model.safetensors`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`
- `vocab.json`

### `raid_model/`
- `.gitattributes`
- `README.md`
- `config.json`
- `model.safetensors`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`

## Verify deployment

```bash
ls -lah meld_model
ls -lah tmr_model
ls -lah raid_model
python3 deploy_meld.py --target-dir ./meld_model
python3 deploy_meld.py --model-id Oxidane/tmr-ai-text-detector --target-dir ./tmr_model
python3 deploy_meld.py --model-id GeorgeDrayson/modernbert-ai-detection-raid-mage --target-dir ./raid_model
```

The second run should report files as "already up to date."
