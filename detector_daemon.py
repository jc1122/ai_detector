#!/usr/bin/env python3
"""JSONL daemon for scoring text with preloaded AI detector models."""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import run_ensemble


ExpertName = str
ALL_EXPERTS: tuple[ExpertName, ...] = ("meld", "tmr", "raid")


def _parse_experts(value: str) -> tuple[ExpertName, ...]:
    experts = tuple(expert.strip().lower() for expert in value.split(",") if expert.strip())
    if not experts:
        raise argparse.ArgumentTypeError("At least one expert must be specified, e.g. meld,tmr,raid")

    for expert in experts:
        if expert not in {"meld", "tmr", "raid"}:
            raise argparse.ArgumentTypeError(
                f"Unsupported expert '{expert}'. Use one or more of: meld, tmr, raid."
            )

    # Preserve order, keep unique entries.
    return tuple(OrderedDict((expert, None) for expert in experts).keys())


def _parse_positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _parse_non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _to_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a numeric value.") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{name} must be a finite value.")
    return parsed


def _parse_weights_string(value: str) -> dict[str, float]:
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise RuntimeError("weights must be two or three comma-separated numeric values.") from exc

    if any(not math.isfinite(weight) for weight in values):
        raise RuntimeError("weights must contain only finite values.")

    parsed = run_ensemble._parse_weights(value)
    return parsed


def _parse_weights_cli(value: str) -> dict[str, float]:
    try:
        return _parse_weights_string(value)
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite.")
    if not 0 <= value <= 1:
        raise RuntimeError(f"{name} must be in [0, 1].")
    return value


def _validate_weight_sum_to_one(weights: dict[str, float], *, context: str) -> None:
    total = sum(weights.values())
    for name, value in weights.items():
        if not math.isfinite(value):
            raise RuntimeError(f"{context} must contain finite values for '{name}'.")
        if value < 0:
            raise RuntimeError(f"{context} must contain non-negative values.")
    if not math.isfinite(total):
        raise RuntimeError(f"{context} must have finite values.")
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"{context} must sum to 1.0.")


def _normalize_weight_triplet(raw_weights: dict[ExpertName, float], *, strict_keys: bool = True) -> dict[str, float]:
    if strict_keys and set(raw_weights.keys()) != set(ALL_EXPERTS):
        raise RuntimeError("weights dict must contain exactly meld, tmr, raid keys.")

    for name in ALL_EXPERTS:
        if raw_weights[name] < 0:
            raise RuntimeError("weights must be non-negative")

    total = sum(raw_weights[name] for name in ALL_EXPERTS)
    if total <= 0:
        raise RuntimeError("weights must sum to a value > 0.")

    return {
        "meld": raw_weights["meld"] / total,
        "tmr": raw_weights["tmr"] / total,
        "raid": raw_weights["raid"] / total,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a preloaded JSONL inference daemon.")
    parser.add_argument("--meld-dir", default="meld_model", help="Path to the MELD checkpoint directory.")
    parser.add_argument("--tmr-dir", default="tmr_model", help="Path to the TMR checkpoint directory.")
    parser.add_argument("--raid-dir", default="raid_model", help="Path to the MAGE ModernBERT checkpoint directory.")
    parser.add_argument(
        "--experts",
        type=_parse_experts,
        default="meld,tmr,raid",
        help="Comma-separated experts to preload: meld,tmr,raid.",
    )
    parser.add_argument(
        "--weights",
        default="0.34,0.33,0.33",
        type=_parse_weights_cli,
        help="Default weights for scoring requests (meld,tmr,raid).",
    )
    parser.add_argument(
        "--threshold",
        type=run_ensemble._parse_probability,
        default=0.5,
        help="Default decision threshold on AI probability.",
    )
    parser.add_argument(
        "--overlap",
        type=_parse_non_negative,
        default=128,
        help="Default token overlap for chunked scoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_positive,
        default=8,
        help="Default batch size for chunk scoring.",
    )
    parser.add_argument(
        "--max-chunks",
        type=_parse_positive,
        default=None,
        help="Default max chunks per expert.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid remote loads and use local files only.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Run on cuda if available with 'auto' (default cpu).",
    )
    parser.add_argument(
        "--threads",
        type=_parse_positive,
        default=None,
        help="Set torch.set_num_threads().",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress third-party stderr output during scoring.",
    )
    return parser.parse_args(argv)


def _normalize_weights_from_request(
    request_weights: object,
    default_weights: dict[str, float],
    loaded_experts: tuple[ExpertName, ...],
) -> dict[str, float]:
    default_request = request_weights is None

    if request_weights is None:
        parsed_weights = {
            "meld": default_weights["meld"],
            "tmr": default_weights["tmr"],
            "raid": default_weights["raid"],
        }
    elif isinstance(request_weights, str):
        parsed_weights = _parse_weights_string(request_weights)
    elif isinstance(request_weights, Sequence) and not isinstance(request_weights, (bytes, str)):
        values = [_to_finite_float(item, name=f"weights[{index}]") for index, item in enumerate(request_weights)]
        if len(values) == 2:
            values.append(0.0)
        if len(values) != 3:
            raise RuntimeError("weights must be two or three numbers")
        if any(value < 0 for value in values):
            raise RuntimeError("weights must be non-negative")
        parsed_weights = {
            "meld": values[0],
            "tmr": values[1],
            "raid": values[2],
        }
    elif isinstance(request_weights, dict):
        if set(request_weights.keys()) != set(ALL_EXPERTS):
            raise RuntimeError("weights dict must contain exactly meld, tmr, raid keys.")
        parsed_weights = {
            expert: _to_finite_float(request_weights[expert], name=f"weights.{expert}")
            for expert in ALL_EXPERTS
        }
    else:
        raise RuntimeError("weights must be a string (e.g. '1,1,1') or a list/tuple.")

    if default_request:
        loaded_set = set(loaded_experts)
        subset = {
            expert: parsed_weights[expert] for expert in ALL_EXPERTS if expert in loaded_set
        }
        subset_total = sum(subset.values())
        if subset_total <= 0:
            raise RuntimeError("Default request weights assign zero to all preloaded experts.")
        return {
            "meld": subset.get("meld", 0.0) / subset_total if "meld" in subset else 0.0,
            "tmr": subset.get("tmr", 0.0) / subset_total if "tmr" in subset else 0.0,
            "raid": subset.get("raid", 0.0) / subset_total if "raid" in subset else 0.0,
        }

    parsed_weights = _normalize_weight_triplet(parsed_weights)

    loaded_set = set(loaded_experts)
    missing = [name for name, value in parsed_weights.items() if value > 0 and name not in loaded_set]
    if missing:
        quoted = ", ".join(f"{name!r}" for name in missing)
        raise RuntimeError(
            f"Explicit request references non-preloaded expert(s): {quoted}. "
            f"Loaded experts: {', '.join(sorted(loaded_set))}."
        )
    return parsed_weights

def _to_float_inclusive_0_1(value: object, *, name: str) -> float:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric in [0, 1].") from exc
    if not 0 <= parsed <= 1:
        raise RuntimeError(f"{name} must be in [0, 1].")
    return parsed


def _to_non_negative_int(value: object, *, name: str) -> int:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be >= 0.")
    return parsed


def _to_positive_int(value: object, *, name: str) -> int:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be > 0.")
    return parsed


def _parse_request_bool(value: object, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise RuntimeError(f"{name} must be a boolean when provided in a request JSON.")


@dataclass
class _ExpertState:
    model: object
    tokenizer: object
    max_length: int
    run_model: Callable[..., object]


class DetectorDaemon:
    """Maintains loaded experts and processes line-delimited JSON requests."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._torch = run_ensemble._ensure_torch()
        if args.threads is not None:
            self._torch.set_num_threads(args.threads)
        self._thread_count = args.threads or getattr(self._torch, "get_num_threads", lambda: None)()

        self._device = self._resolve_device(args.device)
        self._experts: dict[str, _ExpertState] = {}
        self._load_requested_experts()

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def loaded_experts(self) -> tuple[str, ...]:
        return tuple(self._experts.keys())

    @property
    def thread_count(self) -> int | None:
        return self._thread_count

    def _resolve_device(self, raw_device: str) -> object:
        if raw_device == "auto":
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        if raw_device == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Use --device cpu or use --device auto."
            )
        return self._torch.device(raw_device)

    def _load_requested_experts(self) -> None:
        loaders = {
            "meld": run_ensemble.load_meld,
            "tmr": run_ensemble.load_tmr,
            "raid": run_ensemble.load_raid,
        }
        model_dirs = {
            "meld": Path(self.args.meld_dir),
            "tmr": Path(self.args.tmr_dir),
            "raid": Path(self.args.raid_dir),
        }
        for name in self.args.experts:
            load_model = loaders[name]
            model, tokenizer, max_length = run_ensemble._load_expert_model(
                name,
                load_model,
                model_dirs[name],
                local_files_only=self.args.local_files_only,
            )

            if hasattr(model, "to"):
                model.to(self._device)
            if hasattr(model, "eval"):
                model.eval()

            if name == "meld":
                run_model = model.__call__
            else:
                run_model = lambda *, input_ids, attention_mask, _model=model: _model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

            self._experts[name] = _ExpertState(
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                run_model=run_model,
            )

    def _build_payload(self, *, text: str, weights: dict[str, float], threshold: float, overlap: int, batch_size: int, max_chunks: int | None, quiet: bool) -> dict[str, object]:
        expert_names = [name for name, weight in weights.items() if weight > 0]
        if not expert_names:
            raise RuntimeError("No active experts to run. Provide at least one positive weight.")

        _validate_weight_sum_to_one(weights, context="Response weights")

        expert_results = {name: None for name in ("meld", "tmr", "raid")}
        scorer = run_ensemble._score_with_model

        run_scoring: Callable[[], None]

        def execute() -> None:
            for name in expert_names:
                if name not in self._experts:
                    raise RuntimeError(f"Requested expert '{name}' is not preloaded.")
                expert = self._experts[name]
                expert_results[name] = scorer(
                    expert.run_model,
                    expert.tokenizer,
                    text,
                    max_length=expert.max_length,
                    torch_module=self._torch,
                    device=self._device,
                    overlap=overlap,
                    batch_size=batch_size,
                    max_chunks=max_chunks,
                )

        if quiet:
            with run_ensemble._suppress_stderr_fd():
                execute()
        else:
            execute()

        expert_probabilities = {}
        for name in expert_names:
            expert_probabilities[name] = _validate_probability(
                expert_results[name].ai_probability, name=f"{name} ai_probability"
            )

        weighted_ai_probability = _validate_probability(
            sum(weights[name] * expert_probabilities[name] for name in expert_names),
            name="ensemble ai_probability",
        )
        human_probability = 1.0 - weighted_ai_probability
        human_probability = _validate_probability(human_probability, name="ensemble human_probability")
        label = "ai" if weighted_ai_probability >= threshold else "human"

        experts_payload = {
            "meld": run_ensemble._score_payload(expert_results["meld"]),
            "tmr": run_ensemble._score_payload(expert_results["tmr"]),
            "raid": run_ensemble._score_payload(expert_results["raid"]),
        }

        return {
            "text_preview": text[:250],
            "weights": weights,
            "experts": experts_payload,
            "ensemble": {
                "ai_score": weighted_ai_probability,
                "human_score": human_probability,
                "ai_probability": weighted_ai_probability,
                "human_probability": human_probability,
                "threshold": threshold,
                "label": label,
            },
            "calibration": {
                "status": "uncalibrated",
                "calibrated": False,
                "message": "Scores are uncalibrated raw model probabilities. "
                "Provide and use a calibrated model to get calibrated scores.",
            },
            "device": self.device,
        }

    def _handle_request(self, request: dict[str, object]) -> dict[str, object]:
        command = request.get("command")
        if isinstance(command, str):
            if command == "health":
                return {
                    "status": "ok",
                    "command": "health",
                    "loaded_experts": list(self._experts.keys()),
                    "device": self.device,
                    "threads": self.thread_count,
                    "local_files_only": self.args.local_files_only,
                }
            if command == "shutdown":
                self.unload()
                return {
                    "status": "ok",
                    "command": "shutdown",
                    "ack": True,
                    "loaded_experts": [],
                }
            raise RuntimeError(f"Unsupported command '{command}'.")

        text = request.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Each scoring request must include a string `text` field.")
        if not text:
            raise RuntimeError("No input text provided.")

        weights = _normalize_weights_from_request(
            request.get("weights"),
            self.args.weights,
            self.args.experts,
        )
        threshold = _to_float_inclusive_0_1(request.get("threshold", self.args.threshold), name="threshold")
        if threshold is None:
            threshold = self.args.threshold
        overlap = _to_non_negative_int(request.get("overlap", self.args.overlap), name="overlap")
        batch_size = _to_positive_int(request.get("batch_size", self.args.batch_size), name="batch_size")
        max_chunks = request.get("max_chunks", self.args.max_chunks)
        if max_chunks is not None:
            max_chunks = _to_positive_int(max_chunks, name="max_chunks")
        quiet = _parse_request_bool(request.get("quiet"), default=self.args.quiet, name="quiet")

        return self._build_payload(
            text=text,
            weights=weights,
            threshold=threshold,
            overlap=overlap,
            batch_size=batch_size,
            max_chunks=max_chunks,
            quiet=quiet,
        )

    def _handle_line(self, line: str) -> dict[str, object] | None:
        if not line.strip():
            return None
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"error": "Invalid JSON request.", "message": str(exc)}
        if not isinstance(request, dict):
            return {"error": "Request payload must be a JSON object."}

        try:
            return self._handle_request(request)
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "type": exc.__class__.__name__,
            }

    def serve(self, *, stream_in=sys.stdin, stream_out=sys.stdout) -> None:
        for line in stream_in:
            response = self._handle_line(line)
            if response is None:
                continue
            try:
                serialized = json.dumps(response, allow_nan=False)
            except ValueError:
                response = {
                    "error": "Response contains non-finite numbers and could not be serialized.",
                    "type": "RuntimeError",
                }
                serialized = json.dumps(response, allow_nan=False)
            stream_out.write(serialized + "\n")
            stream_out.flush()
            if response.get("command") == "shutdown":
                return

    def unload(self) -> None:
        for expert in self._experts.values():
            if hasattr(expert.model, "cpu"):
                try:
                    expert.model.cpu()
                except Exception:
                    pass
            if hasattr(expert.model, "to"):
                try:
                    expert.model.to("cpu")
                except Exception:
                    pass
        self._experts.clear()
        self._torch.cuda.empty_cache() if hasattr(self._torch, "cuda") and hasattr(self._torch.cuda, "empty_cache") else None
        import gc

        gc.collect()


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        daemon = DetectorDaemon(args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    daemon.serve()


if __name__ == "__main__":
    main()
