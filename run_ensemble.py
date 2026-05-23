#!/usr/bin/env python3
"""Run MELD, TMR, and MAGE ModernBERT detectors as an ensemble on text."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ExpertResult:
    ai_probability: float
    chunks: int


class MELDDetector(nn.Module):
    def __init__(self, backbone: str, n_generators: int, n_attacks: int, n_domains: int, *, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_main = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.head_gen = nn.Linear(hidden_size, n_generators)
        self.head_att = nn.Linear(hidden_size, n_attacks)
        self.head_dom = nn.Linear(hidden_size, n_domains)
        self.log_var_main = nn.Parameter(torch.zeros(()))
        self.log_var_gen = nn.Parameter(torch.zeros(()))
        self.log_var_att = nn.Parameter(torch.zeros(()))
        self.log_var_dom = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return self.head_main(pooled).float()


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return sys.stdin.read().strip()


def _parse_weights(value: str) -> dict[str, float]:
    values = [float(part.strip()) for part in value.split(",")]
    if len(values) == 2:
        # Backward-compatible: keep previous 2-model behaviour with zero weight for MAGE.
        values.append(0.0)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected two or three comma-separated values: meld,tmr,(raid)")
    if values[0] < 0 or values[1] < 0 or values[2] < 0:
        raise argparse.ArgumentTypeError("weights must be non-negative")
    total = values[0] + values[1] + values[2]
    if total == 0:
        raise argparse.ArgumentTypeError("weights must sum to a value > 0")
    return {
        "meld": values[0] / total,
        "tmr": values[1] / total,
        "raid": values[2] / total,
    }


def _score_with_model(
    run_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    device: torch.device,
    *,
    overlap: int,
) -> ExpertResult:
    probabilities: list[float] = []
    if not text:
        raise RuntimeError("No input text provided.")
    if not max_length or max_length <= 0:
        raise RuntimeError(f"Invalid model max_length={max_length}")

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=min(max(1, overlap), max_length - 1),
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    chunk_input_ids = tokenized["input_ids"]
    chunk_attention_masks = tokenized["attention_mask"]

    chunk_count = len(chunk_input_ids)
    with torch.no_grad():
        for i in range(chunk_count):
            chunk_ids = torch.tensor([chunk_input_ids[i]], dtype=torch.long, device=device)
            chunk_attention = torch.tensor([chunk_attention_masks[i]], dtype=torch.long, device=device)
            logits = run_model(input_ids=chunk_ids, attention_mask=chunk_attention)
            probs = torch.softmax(logits.float(), dim=-1)[0]
            probabilities.append(float(probs[1].item()))

    if not probabilities:
        raise RuntimeError("No chunks were scored. Check tokenization and input text.")
    return ExpertResult(ai_probability=sum(probabilities) / len(probabilities), chunks=chunk_count)


def load_meld(model_dir: Path) -> tuple[callable, AutoTokenizer, int]:
    cfg = json.loads((model_dir / "meld_config.json").read_text(encoding="utf-8"))
    model = MELDDetector(
        cfg["backbone"],
        cfg["n_generators"],
        cfg["n_attacks"],
        cfg["n_domains"],
        num_labels=cfg.get("num_labels", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model.eval(), tokenizer, int(cfg["max_length"])


def load_tmr(model_dir: Path) -> tuple[callable, AutoTokenizer, int]:
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_position_embeddings = int(getattr(config, "max_position_embeddings", tokenizer.model_max_length))
    max_length = max_position_embeddings - 2 if max_position_embeddings > 2 else tokenizer.model_max_length
    model.eval()
    return model.eval(), tokenizer, max_length


def load_raid(model_dir: Path) -> tuple[callable, AutoTokenizer, int]:
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_position_embeddings = int(getattr(config, "max_position_embeddings", tokenizer.model_max_length))
    max_length = max_position_embeddings - 2 if max_position_embeddings > 2 else tokenizer.model_max_length
    model.eval()
    return model.eval(), tokenizer, max_length


def run_ensemble(text: str, args: argparse.Namespace) -> dict[str, object]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if not text:
        raise RuntimeError("No input text provided.")

    meld_model, meld_tokenizer, meld_max_length = load_meld(Path(args.meld_dir))
    tmr_model, tmr_tokenizer, tmr_max_length = load_tmr(Path(args.tmr_dir))
    raid_model, raid_tokenizer, raid_max_length = load_raid(Path(args.raid_dir))

    meld_model.to(device)
    tmr_model.to(device)
    raid_model.to(device)

    experts = {
        "meld": _score_with_model(
            meld_model.__call__,
            meld_tokenizer,
            text,
            max_length=meld_max_length,
            device=device,
            overlap=args.overlap,
        ),
        "tmr": _score_with_model(
            lambda *, input_ids, attention_mask: tmr_model(input_ids=input_ids, attention_mask=attention_mask).logits,
            tmr_tokenizer,
            text,
            max_length=tmr_max_length,
            device=device,
            overlap=args.overlap,
        ),
        "raid": _score_with_model(
            lambda *, input_ids, attention_mask: raid_model(input_ids=input_ids, attention_mask=attention_mask).logits,
            raid_tokenizer,
            text,
            max_length=raid_max_length,
            device=device,
            overlap=args.overlap,
        ),
    }

    score = (
        args.weights["meld"] * experts["meld"].ai_probability
        + args.weights["tmr"] * experts["tmr"].ai_probability
        + args.weights["raid"] * experts["raid"].ai_probability
    )
    label = "ai" if score >= args.threshold else "human"

    return {
        "text_preview": text[:250],
        "weights": args.weights,
        "experts": {
            "meld": {
                "ai_probability": experts["meld"].ai_probability,
                "chunks": experts["meld"].chunks,
            },
            "tmr": {
                "ai_probability": experts["tmr"].ai_probability,
                "chunks": experts["tmr"].chunks,
            },
            "raid": {
                "ai_probability": experts["raid"].ai_probability,
                "chunks": experts["raid"].chunks,
            },
        },
        "ensemble": {
            "ai_probability": score,
            "threshold": args.threshold,
            "label": label,
        },
        "device": str(device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MELD, TMR, and MAGE ModernBERT models as an ensemble.")
    parser.add_argument("--meld-dir", default="meld_model", help="Path to the MELD checkpoint directory.")
    parser.add_argument("--tmr-dir", default="tmr_model", help="Path to the TMR checkpoint directory.")
    parser.add_argument("--raid-dir", default="raid_model", help="Path to the MAGE ModernBERT checkpoint directory.")
    parser.add_argument("--text", help="Input text as CLI argument.")
    parser.add_argument("--text-file", dest="text_file", help="Read input text from a file.")
    parser.add_argument(
        "--weights",
        default="0.34,0.33,0.33",
        type=_parse_weights,
        help="Ensemble weights as two or three comma-separated values: meld,tmr,(raid).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on AI probability.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Token overlap for chunked scoring of long texts.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Run on cuda if available, otherwise cpu.",
    )
    parser.add_argument("--json", action="store_true", help="Print compact JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = _read_text(args)
    result = run_ensemble(text, args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("MELD AI probability:", f"{result['experts']['meld']['ai_probability']:.6f}")
        print("TMR  AI probability:", f"{result['experts']['tmr']['ai_probability']:.6f}")
        print("RAID AI probability:", f"{result['experts']['raid']['ai_probability']:.6f}")
        print(
            "Ensemble AI probability:",
            f"{result['ensemble']['ai_probability']:.6f} "
            f"(threshold {result['ensemble']['threshold']:.2f})",
        )
        print("Decision:", result["ensemble"]["label"])
        print(
            "Chunks:",
            f"meld={result['experts']['meld']['chunks']}",
            f"tmr={result['experts']['tmr']['chunks']}",
            f"raid={result['experts']['raid']['chunks']}",
        )


if __name__ == "__main__":
    main()
