#!/usr/bin/env python3
"""Shared calibration config loading for detector CLIs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALL_EXPERTS: tuple[str, ...] = ("meld", "tmr", "raid")

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "pl-technical-ood": {
        "schema_version": 1,
        "id": "pl_technical_ood_2026_05_24",
        "description": "Built-in Polish technical/OOD operating-point calibration.",
        "method": "grid_search_fixed_model_weights_heuristic_weight_midpoint_threshold",
        "source_result_file": "data/evaluation/polish_pre2020_technical_papers/broad_eval_windows_2026-05-24.json",
        "source_calibration_file": "data/evaluation/polish_pre2020_technical_papers/calibration_pl_ood_2026-05-24.json",
        "weights": {"meld": 0.75, "tmr": 0.0, "raid": 0.25},
        "heuristic_weight": 0.6,
        "threshold": 0.4345518078804016,
        "applies_to": {
            "language": "pl",
            "domain": "technical_scientific_ood",
            "notes": "Source-group split calibration; not probability calibration.",
        },
    }
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Validated operating-point calibration metadata."""

    source_path: Path
    schema_version: int
    calibration_id: str
    weights: dict[str, float]
    threshold: float
    heuristic_weight: float
    method: str
    description: str
    payload: dict[str, Any]


def _to_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{name} must be finite.")
    return parsed


def normalize_weights(raw_weights: object) -> dict[str, float]:
    """Validate and normalize MELD/TMR/RAID weights from JSON-compatible data."""

    if isinstance(raw_weights, dict):
        if set(raw_weights) != set(ALL_EXPERTS):
            raise RuntimeError("calibration weights must contain exactly meld, tmr, raid keys.")
        weights = {
            expert: _to_finite_float(raw_weights[expert], name=f"weights.{expert}")
            for expert in ALL_EXPERTS
        }
    elif isinstance(raw_weights, list):
        if len(raw_weights) != 3:
            raise RuntimeError("calibration weights list must contain three values: meld, tmr, raid.")
        weights = {
            expert: _to_finite_float(raw_weights[index], name=f"weights[{index}]")
            for index, expert in enumerate(ALL_EXPERTS)
        }
    else:
        raise RuntimeError("calibration weights must be an object or list.")

    if any(value < 0 for value in weights.values()):
        raise RuntimeError("calibration weights must be non-negative.")
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError("calibration weights must sum to a value > 0.")

    return {expert: weights[expert] / total for expert in ALL_EXPERTS}


def validate_threshold(value: object) -> float:
    threshold = _to_finite_float(value, name="threshold")
    if not 0 <= threshold <= 1:
        raise RuntimeError("calibration threshold must be in [0, 1].")
    return threshold


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    """Load a JSON calibration config used to set default weights/threshold."""

    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read calibration file '{source_path}': {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Calibration file '{source_path}' is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("calibration file must contain a JSON object.")

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise RuntimeError("unsupported calibration schema_version; expected 1.")

    calibration_id = str(payload.get("id", "")).strip()
    if not calibration_id:
        raise RuntimeError("calibration file must include a non-empty id.")

    weights = normalize_weights(payload.get("weights"))
    threshold = validate_threshold(payload.get("threshold"))
    heuristic_weight = validate_threshold(payload.get("heuristic_weight", 0.0))
    method = str(payload.get("method", "operating_point_grid_search")).strip()
    description = str(payload.get("description", "")).strip()

    return CalibrationConfig(
        source_path=source_path,
        schema_version=schema_version,
        calibration_id=calibration_id,
        weights=weights,
        threshold=threshold,
        heuristic_weight=heuristic_weight,
        method=method,
        description=description,
        payload=payload,
    )


def load_builtin_profile(name: str) -> CalibrationConfig:
    try:
        payload = BUILTIN_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_PROFILES))
        raise RuntimeError(f"Unknown calibration profile '{name}'. Available profiles: {known}.") from exc

    weights = normalize_weights(payload["weights"])
    threshold = validate_threshold(payload["threshold"])
    heuristic_weight = validate_threshold(payload.get("heuristic_weight", 0.0))
    return CalibrationConfig(
        source_path=Path(str(payload.get("source_calibration_file", f"<builtin:{name}>"))),
        schema_version=int(payload["schema_version"]),
        calibration_id=str(payload["id"]),
        weights=weights,
        threshold=threshold,
        heuristic_weight=heuristic_weight,
        method=str(payload["method"]),
        description=str(payload.get("description", "")),
        payload=dict(payload),
    )


def calibration_payload(
    config: CalibrationConfig | None,
    *,
    applied_to_weights: bool = False,
    applied_to_threshold: bool = False,
    applied_to_heuristic_weight: bool = False,
) -> dict[str, object]:
    """Return the public output-contract calibration section."""

    if config is None:
        return {
            "status": "uncalibrated",
            "calibrated": False,
            "message": "Scores are uncalibrated raw model probabilities. "
            "Provide and use a calibrated model to get calibrated scores.",
        }

    calibrated = applied_to_weights or applied_to_threshold or applied_to_heuristic_weight
    return {
        "status": "operating_point_calibrated" if calibrated else "calibration_available_not_applied",
        "calibrated": calibrated,
        "probability_calibrated": False,
        "id": config.calibration_id,
        "method": config.method,
        "source_path": str(config.source_path),
        "applied_to_weights": applied_to_weights,
        "applied_to_threshold": applied_to_threshold,
        "applied_to_heuristic_weight": applied_to_heuristic_weight,
        "message": "Weights and threshold come from a local validation artifact. "
        "Expert probabilities remain raw model scores, not calibrated probabilities.",
    }
