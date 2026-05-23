#!/usr/bin/env python3
"""Run MELD, TMR, and MAGE ModernBERT detectors as an ensemble on text."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol


def _ensure_torch() -> object:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to run inference. Install it before running without --help."
        ) from exc


def _ensure_transformers() -> tuple[object, object, object]:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to run inference. Install it first.") from exc

    return AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _ensure_meld_dependencies() -> tuple[type, callable]:
    torch = _ensure_torch()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load MELD checkpoints.") from exc

    import torch.nn as nn

    class MELDDetector(nn.Module):
        def __init__(
            self,
            backbone: str,
            n_generators: int,
            n_attacks: int,
            n_domains: int,
            *,
            backbone_model: "torch.nn.Module",
            num_labels: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.backbone = backbone_model
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

        def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = self.dropout(pooled)
            return self.head_main(pooled).float()

    return MELDDetector, load_file


def _load_meld_backbone(backbone: str, _model_dir: Path, *, local_files_only: bool) -> object:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the MELD backbone.") from exc

    try:
        return AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                "Cannot load MELD backbone in local-only mode. The backbone must be available in the local Hugging Face cache "
                "or run without --local-files-only to allow downloading it."
            ) from exc
        raise RuntimeError(
            "Failed to load MELD backbone. Check model availability and network/disk accessibility."
        ) from exc


@dataclass(frozen=True)
class ExpertResult:
    ai_probability: float
    chunks: int


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        text_path = Path(args.text_file)
        try:
            return text_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read --text-file '{text_path}': {exc.strerror or exc}"
            ) from exc
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


def _parse_probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return parsed


def _parse_non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _parse_positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _score_with_model(
    run_model: Callable[..., object],
    tokenizer: object,
    text: str,
    max_length: int,
    torch_module: object,
    device: "torch_device",
    *,
    overlap: int,
    batch_size: int,
    max_chunks: int | None,
) -> ExpertResult:
    probabilities: list[float] = []
    if not text:
        raise RuntimeError("No input text provided.")
    if not max_length or max_length <= 0:
        raise RuntimeError(f"Invalid model max_length={max_length}")

    stride = min(overlap, max_length - 1)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    chunk_input_ids = tokenized["input_ids"]
    chunk_attention_masks = tokenized["attention_mask"]
    chunk_count = len(chunk_input_ids)

    if max_chunks is not None:
        chunk_count = min(chunk_count, max_chunks)
        chunk_input_ids = chunk_input_ids[:chunk_count]
        chunk_attention_masks = chunk_attention_masks[:chunk_count]

    if not chunk_input_ids:
        raise RuntimeError("No chunks were scored. Check tokenization and input text.")

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    with torch_module.inference_mode():
        for start in range(0, chunk_count, batch_size):
            end = min(start + batch_size, chunk_count)
            batch_ids = chunk_input_ids[start:end]
            batch_attention = chunk_attention_masks[start:end]
            max_len = max(len(tokens) for tokens in batch_ids)
            batch_ids = [
                tokens + [pad_token_id] * (max_len - len(tokens))
                for tokens in batch_ids
            ]
            batch_attention = [
                mask + [0] * (max_len - len(mask))
                for mask in batch_attention
            ]

            chunk_ids = torch_module.tensor(
                batch_ids,
                dtype=torch_module.long,
                device=device,
            )
            chunk_attention = torch_module.tensor(
                batch_attention,
                dtype=torch_module.long,
                device=device,
            )
            logits = run_model(input_ids=chunk_ids, attention_mask=chunk_attention)
            probs = torch_module.softmax(logits.float(), dim=-1)[:, 1]
            probabilities.extend(float(value.item()) for value in probs)

    if not probabilities:
        raise RuntimeError("No chunks were scored. Check tokenization and input text.")
    return ExpertResult(ai_probability=sum(probabilities) / len(probabilities), chunks=chunk_count)


def load_meld(model_dir: Path, local_files_only: bool = False) -> tuple[Callable[[], object], object, int]:
    MELDDetector, load_file = _ensure_meld_dependencies()
    cfg = json.loads((model_dir / "meld_config.json").read_text(encoding="utf-8"))
    backbone_model = _load_meld_backbone(cfg["backbone"], model_dir, local_files_only=local_files_only)
    model = MELDDetector(
        cfg["backbone"],
        cfg["n_generators"],
        cfg["n_attacks"],
        cfg["n_domains"],
        backbone_model=backbone_model,
        num_labels=cfg.get("num_labels", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
    return model.eval(), tokenizer, int(cfg["max_length"])


def load_tmr(model_dir: Path, local_files_only: bool = False) -> tuple[Callable[[], object], object, int]:
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer = _ensure_transformers()
    config = AutoConfig.from_pretrained(model_dir, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, config=config, local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
    max_position_embeddings = int(getattr(config, "max_position_embeddings", tokenizer.model_max_length))
    max_length = max_position_embeddings - 2 if max_position_embeddings > 2 else tokenizer.model_max_length
    model.eval()
    return model.eval(), tokenizer, max_length


def load_raid(model_dir: Path, local_files_only: bool = False) -> tuple[Callable[[], object], object, int]:
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer = _ensure_transformers()
    config = AutoConfig.from_pretrained(model_dir, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, config=config, local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
    max_position_embeddings = int(getattr(config, "max_position_embeddings", tokenizer.model_max_length))
    max_length = max_position_embeddings - 2 if max_position_embeddings > 2 else tokenizer.model_max_length
    model.eval()
    return model.eval(), tokenizer, max_length


def _score_payload(result: ExpertResult | None) -> dict[str, object]:
    if result is None:
        return {
            "ai_score": None,
            "human_score": None,
            "ai_probability": None,
            "human_probability": None,
            "chunks": 0,
            "loaded": False,
            "notes": "Not scored because model weight is 0.0.",
        }

    ai_score = result.ai_probability
    return {
        "ai_score": ai_score,
        "human_score": 1.0 - ai_score,
        "ai_probability": ai_score,
        "human_probability": 1.0 - ai_score,
        "chunks": result.chunks,
        "loaded": True,
    }


@contextlib.contextmanager
def _suppress_stderr_fd() -> Iterator[None]:
    original_stderr_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)


def _format_plain_score(score: float | None) -> str:
    if score is None:
        return "skipped"
    return f"{score:.6f}"


class _ExpertLoader(Protocol):
    def __call__(
        self,
        model_dir: Path,
        *,
        local_files_only: bool,
    ) -> tuple[Callable[[], object], object, int]:
        ...


def _load_expert_model(
    expert_name: str,
    loader: _ExpertLoader,
    model_dir: Path,
    *,
    local_files_only: bool,
) -> tuple[Callable[[], object], object, int]:
    try:
        return loader(model_dir, local_files_only=local_files_only)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to initialize model for expert '{expert_name}' due to a missing dependency: {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Failed to initialize model for expert '{expert_name}' from '{model_dir}': {exc}"
        ) from exc


def run_ensemble(text: str, args: argparse.Namespace) -> dict[str, object]:
    torch = _ensure_torch()

    if not text:
        raise RuntimeError("No input text provided.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch."
        )
    else:
        device = torch.device(args.device)

    model_dir_by_name = {
        "meld": Path(args.meld_dir),
        "tmr": Path(args.tmr_dir),
        "raid": Path(args.raid_dir),
    }
    loader_by_name = {"meld": load_meld, "tmr": load_tmr, "raid": load_raid}

    expert_names = [name for name, weight in args.weights.items() if weight > 0]
    if not expert_names:
        raise RuntimeError("No active experts to run. Provide at least one positive weight.")

    expert_results: dict[str, ExpertResult | None] = {name: None for name in model_dir_by_name}

    for name in expert_names:
        load_model = loader_by_name[name]
        model, tokenizer, max_length = _load_expert_model(
            name,
            load_model,
            model_dir_by_name[name],
            local_files_only=args.local_files_only,
        )
        model.to(device)
        model.eval()

        run_model: Callable[..., object] = (
            model.__call__
            if name == "meld"
            else lambda *, input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits
        )

        expert_results[name] = _score_with_model(
            run_model,
            tokenizer,
            text,
            max_length=max_length,
            torch_module=torch,
            device=device,
            overlap=args.overlap,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
        )

    ai_probability = sum(
        args.weights[name] * expert_results[name].ai_probability for name in expert_names
    )
    human_probability = 1.0 - ai_probability
    label = "ai" if ai_probability >= args.threshold else "human"

    return {
        "text_preview": text[:250],
        "weights": args.weights,
        "experts": {
            "meld": _score_payload(expert_results["meld"]),
            "tmr": _score_payload(expert_results["tmr"]),
            "raid": _score_payload(expert_results["raid"]),
        },
        "ensemble": {
            "ai_score": ai_probability,
            "human_score": human_probability,
            "ai_probability": ai_probability,
            "human_probability": human_probability,
            "threshold": args.threshold,
            "label": label,
        },
        "calibration": {
            "status": "uncalibrated",
            "calibrated": False,
            "message": "Scores are uncalibrated raw model probabilities. "
            "Provide and use a calibrated model to get calibrated scores.",
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
        type=_parse_probability,
        default=0.5,
        help="Decision threshold on AI probability.",
    )
    parser.add_argument(
        "--overlap",
        type=_parse_non_negative,
        default=128,
        help="Token overlap for chunked scoring of long texts.",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_positive,
        default=8,
        help="Batch size for chunk scoring.",
    )
    parser.add_argument(
        "--max-chunks",
        type=_parse_positive,
        default=None,
        help="Optional cap on chunks scored per expert.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid remote loads and use local files only.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Run on cuda if available, otherwise cpu.",
    )
    parser.add_argument("--json", action="store_true", help="Print compact JSON output.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress third-party stderr output during model loading and scoring.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        text = _read_text(args)
        if args.quiet:
            with _suppress_stderr_fd():
                result = run_ensemble(text, args)
        else:
            result = run_ensemble(text, args)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("MELD AI probability:", _format_plain_score(result["experts"]["meld"]["ai_score"]))
    print("TMR  AI probability:", _format_plain_score(result["experts"]["tmr"]["ai_score"]))
    print("RAID AI probability:", _format_plain_score(result["experts"]["raid"]["ai_score"]))
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
