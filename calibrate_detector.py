#!/usr/bin/env python3
"""Fit simple operating-point calibration from stored detector score artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from calibration_config import ALL_EXPERTS, normalize_weights


def _parse_probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _parse_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _parse_weights_arg(value: str) -> dict[str, float]:
    try:
        raw_values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be three comma-separated numeric values.") from exc
    if len(raw_values) != 3:
        raise argparse.ArgumentTypeError("weights must contain three values: meld,tmr,raid.")
    try:
        return normalize_weights(dict(zip(ALL_EXPERTS, raw_values, strict=True)))
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _weighted_score(record: dict[str, object], weights: dict[str, float]) -> float:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError(f"Record {record.get('id', '<unknown>')} is missing scores.")
    return sum(weights[expert] * float(scores[expert]) for expert in ALL_EXPERTS)


def _record_score(record: dict[str, object], weights: dict[str, float], heuristic_weight: float) -> float:
    model_score = _weighted_score(record, weights)
    if heuristic_weight == 0:
        return model_score
    scores = record.get("scores")
    if not isinstance(scores, dict) or "heuristic" not in scores:
        raise RuntimeError(
            f"Record {record.get('id', '<unknown>')} is missing a heuristic score required for hybrid calibration."
        )
    heuristic_score = float(scores["heuristic"])
    return (1.0 - heuristic_weight) * model_score + heuristic_weight * heuristic_score


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-12:
        yield round(value, 10)
        value += step


def _candidate_weights(step: float) -> Iterable[dict[str, float]]:
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("grid step must divide 1.0 evenly.")
    for meld_units in range(units + 1):
        for tmr_units in range(units + 1 - meld_units):
            raid_units = units - meld_units - tmr_units
            yield normalize_weights(
                {
                    "meld": meld_units * step,
                    "tmr": tmr_units * step,
                    "raid": raid_units * step,
                }
            )


def _candidate_heuristic_weights(*, fit_heuristic_weight: bool, step: float, max_value: float) -> Iterable[float]:
    if not fit_heuristic_weight:
        yield 0.0
        return
    for value in _frange(0.0, max_value, step):
        yield value
    if not math.isclose(max_value / step, round(max_value / step), rel_tol=0, abs_tol=1e-9):
        yield max_value


def _score_candidate(
    *,
    weights: dict[str, float],
    heuristic_weight: float,
    human_records: list[dict[str, object]],
    ai_records: list[dict[str, object]],
) -> dict[str, object]:
    human_scores = [_record_score(record, weights, heuristic_weight) for record in human_records]
    ai_scores = [_record_score(record, weights, heuristic_weight) for record in ai_records]
    if not human_scores:
        raise RuntimeError("No human calibration records selected.")
    if not ai_scores:
        raise RuntimeError("No AI-control calibration records selected.")

    max_human = max(human_scores)
    min_ai = min(ai_scores)
    threshold = (max_human + min_ai) / 2
    threshold = max(0.0, min(1.0, threshold))
    false_positives = sum(score >= threshold for score in human_scores)
    false_negatives = sum(score < threshold for score in ai_scores)
    margin = min_ai - max_human

    return {
        "weights": weights,
        "heuristic_weight": heuristic_weight,
        "threshold": threshold,
        "margin": margin,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "human": {
            "n": len(human_scores),
            "mean": sum(human_scores) / len(human_scores),
            "max": max_human,
            "min": min(human_scores),
        },
        "ai_control": {
            "n": len(ai_scores),
            "mean": sum(ai_scores) / len(ai_scores),
            "min": min_ai,
            "max": max(ai_scores),
        },
    }


def _evaluate_candidate(
    *,
    weights: dict[str, float],
    heuristic_weight: float,
    threshold: float,
    human_records: list[dict[str, object]],
    ai_records: list[dict[str, object]],
) -> dict[str, object]:
    human_scores = [_record_score(record, weights, heuristic_weight) for record in human_records]
    ai_scores = [_record_score(record, weights, heuristic_weight) for record in ai_records]
    if not human_scores or not ai_scores:
        return {
            "human": {"n": len(human_scores)},
            "ai_control": {"n": len(ai_scores)},
            "false_positives_at_threshold": None,
            "false_negatives_at_threshold": None,
            "margin": None,
        }
    return {
        "human": {
            "n": len(human_scores),
            "mean": sum(human_scores) / len(human_scores),
            "min": min(human_scores),
            "max": max(human_scores),
        },
        "ai_control": {
            "n": len(ai_scores),
            "mean": sum(ai_scores) / len(ai_scores),
            "min": min(ai_scores),
            "max": max(ai_scores),
        },
        "false_positives_at_threshold": sum(score >= threshold for score in human_scores),
        "false_negatives_at_threshold": sum(score < threshold for score in ai_scores),
        "margin": min(ai_scores) - max(human_scores),
    }


def _split_by_source(
    records: list[dict[str, object]],
    *,
    label: str,
    test_fraction: float,
    split_seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if record.get("label") != label:
            continue
        source_id = str(record.get("source_id") or record.get("id") or "")
        if not source_id:
            raise RuntimeError("selected records must include source_id or id for split-aware calibration.")
        groups.setdefault(source_id, []).append(record)

    if len(groups) < 2:
        raise RuntimeError(f"Need at least two source groups for label {label!r}.")

    ordered_sources = sorted(
        groups,
        key=lambda source_id: hashlib.sha256(f"{split_seed}:{label}:{source_id}".encode("utf-8")).hexdigest(),
    )
    test_count = max(1, round(len(ordered_sources) * test_fraction))
    test_count = min(test_count, len(ordered_sources) - 1)
    test_sources = set(ordered_sources[:test_count])

    train = [record for source_id, group in groups.items() for record in group if source_id not in test_sources]
    test = [record for source_id, group in groups.items() for record in group if source_id in test_sources]
    return train, test, sorted(test_sources)


def fit_calibration(
    baseline: dict[str, object],
    *,
    grid_step: float,
    view: str,
    human_quality: str,
    test_fraction: float,
    split_seed: str,
    fixed_weights: dict[str, float] | None = None,
    fit_heuristic_weight: bool = False,
    heuristic_grid_step: float = 0.05,
    max_heuristic_weight: float = 1.0,
) -> dict[str, object]:
    records = baseline.get("records")
    if not isinstance(records, list):
        raise RuntimeError("baseline result must include a records list.")

    selected = [
        record
        for record in records
        if isinstance(record, dict) and (view == "all" or record.get("view") == view)
    ]
    human_records_all = [
        record
        for record in selected
        if record.get("label") == "human_pre2020" and record.get("quality") == human_quality
    ]
    ai_records_all = [record for record in selected if record.get("label") == "synthetic_ai_control"]
    human_train, human_test, human_test_sources = _split_by_source(
        human_records_all,
        label="human_pre2020",
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    ai_train, ai_test, ai_test_sources = _split_by_source(
        ai_records_all,
        label="synthetic_ai_control",
        test_fraction=test_fraction,
        split_seed=split_seed,
    )

    weight_candidates = [fixed_weights] if fixed_weights is not None else list(_candidate_weights(grid_step))
    heuristic_weight_candidates = list(
        _candidate_heuristic_weights(
            fit_heuristic_weight=fit_heuristic_weight,
            step=heuristic_grid_step,
            max_value=max_heuristic_weight,
        )
    )
    candidates = [
        _score_candidate(
            weights=weights,
            heuristic_weight=heuristic_weight,
            human_records=human_train,
            ai_records=ai_train,
        )
        for weights in weight_candidates
        for heuristic_weight in heuristic_weight_candidates
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate["false_positives"],
            candidate["false_negatives"],
            -candidate["margin"],
            candidate["human"]["mean"],
            candidate["heuristic_weight"],
        )
    )
    best = candidates[0]
    weights = best["weights"]
    heuristic_weight = best["heuristic_weight"]
    threshold = best["threshold"]
    return {
        "best": best,
        "top_candidates": candidates[:10],
        "record_selection": {
            "view": view,
            "human_quality": human_quality,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "fixed_weights": fixed_weights,
            "fit_heuristic_weight": fit_heuristic_weight,
            "heuristic_grid_step": heuristic_grid_step,
            "max_heuristic_weight": max_heuristic_weight,
            "human_train_n": len(human_train),
            "human_test_n": len(human_test),
            "human_total_n": len(human_records_all),
            "human_test_sources": human_test_sources,
            "ai_control_train_n": len(ai_train),
            "ai_control_test_n": len(ai_test),
            "ai_control_total_n": len(ai_records_all),
            "ai_control_test_sources": ai_test_sources,
        },
        "metrics": {
            "train": _evaluate_candidate(
                weights=weights,
                heuristic_weight=heuristic_weight,
                threshold=threshold,
                human_records=human_train,
                ai_records=ai_train,
            ),
            "heldout": _evaluate_candidate(
                weights=weights,
                heuristic_weight=heuristic_weight,
                threshold=threshold,
                human_records=human_test,
                ai_records=ai_test,
            ),
            "all": _evaluate_candidate(
                weights=weights,
                heuristic_weight=heuristic_weight,
                threshold=threshold,
                human_records=human_records_all,
                ai_records=ai_records_all,
            ),
        },
    }


def build_config(
    *,
    baseline_path: Path,
    baseline: dict[str, object],
    fit: dict[str, object],
    calibration_id: str,
) -> dict[str, object]:
    best = fit["best"]
    heuristic_weight = float(best.get("heuristic_weight", 0.0))
    method = (
        "grid_search_fixed_model_weights_heuristic_weight_midpoint_threshold"
        if fit["record_selection"]["fixed_weights"] is not None and heuristic_weight
        else "grid_search_model_weights_heuristic_weight_midpoint_threshold"
        if heuristic_weight
        else "grid_search_weights_midpoint_threshold"
    )
    return {
        "schema_version": 1,
        "id": calibration_id,
        "description": "Polish technical/OOD operating-point calibration from pre-2020 human papers "
        "and synthetic AI controls.",
        "method": method,
        "source_result_file": str(baseline_path),
        "applies_to": {
            "language": "pl",
            "domain": "technical_scientific_ood",
            "notes": "Use as an operating-point baseline for Polish technical text; not probability calibration.",
        },
        "weights": best["weights"],
        "heuristic_weight": heuristic_weight,
        "threshold": best["threshold"],
        "metrics": {
            "record_selection": fit["record_selection"],
            "train": fit["metrics"]["train"],
            "heldout": fit["metrics"]["heldout"],
            "all": fit["metrics"]["all"],
        },
        "top_candidates": fit["top_candidates"],
        "source_run": baseline.get("run", {}),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit detector operating-point calibration from stored scores.")
    parser.add_argument("--baseline-result", required=True, help="Path to a stored broad evaluation JSON result.")
    parser.add_argument("--output", required=True, help="Path to write calibration JSON.")
    parser.add_argument("--id", default="pl_technical_ood_2026_05_24", help="Calibration id to write.")
    parser.add_argument("--grid-step", type=_parse_positive_float, default=0.05, help="Weight grid step.")
    parser.add_argument("--view", choices=["all", "short_180w", "long_700w"], default="all")
    parser.add_argument("--human-quality", default="pl_clean", help="Human quality label to fit against.")
    parser.add_argument(
        "--fixed-weights",
        type=_parse_weights_arg,
        help="Use fixed model weights as meld,tmr,raid and fit only threshold/heuristic settings.",
    )
    parser.add_argument(
        "--fit-heuristic-weight",
        action="store_true",
        help="Search a heuristic blend weight using scores.heuristic from the baseline artifact.",
    )
    parser.add_argument(
        "--heuristic-grid-step",
        type=_parse_positive_float,
        default=0.05,
        help="Grid step for --fit-heuristic-weight.",
    )
    parser.add_argument(
        "--max-heuristic-weight",
        type=_parse_probability,
        default=1.0,
        help="Maximum heuristic blend weight considered by --fit-heuristic-weight.",
    )
    parser.add_argument(
        "--test-fraction",
        type=_parse_probability,
        default=0.34,
        help="Source-group fraction held out for reporting.",
    )
    parser.add_argument("--split-seed", default="pl-technical-ood-v1", help="Stable source split seed.")
    parser.add_argument(
        "--max-false-positives",
        type=int,
        default=0,
        help="Fail if the selected candidate exceeds this false-positive count.",
    )
    parser.add_argument(
        "--max-false-negatives",
        type=int,
        default=0,
        help="Fail if the selected candidate exceeds this false-negative count.",
    )
    parser.add_argument(
        "--min-margin",
        type=_parse_probability,
        default=0.05,
        help="Fail if selected min AI-control score minus max human score is smaller.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline_path = Path(args.baseline_result)
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Error: cannot read baseline result '{baseline_path}': {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: baseline result is not valid JSON: {exc}") from exc

    try:
        fit = fit_calibration(
            baseline,
            grid_step=args.grid_step,
            view=args.view,
            human_quality=args.human_quality,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            fixed_weights=args.fixed_weights,
            fit_heuristic_weight=args.fit_heuristic_weight,
            heuristic_grid_step=args.heuristic_grid_step,
            max_heuristic_weight=args.max_heuristic_weight,
        )
        best = fit["best"]
        heldout = fit["metrics"]["heldout"]
        if heldout["false_positives_at_threshold"] > args.max_false_positives:
            raise RuntimeError("selected calibration exceeds heldout false-positive limit.")
        if heldout["false_negatives_at_threshold"] > args.max_false_negatives:
            raise RuntimeError("selected calibration exceeds heldout false-negative limit.")
        if heldout["margin"] < args.min_margin:
            raise RuntimeError("selected calibration heldout margin is below --min-margin.")

        config = build_config(
            baseline_path=baseline_path,
            baseline=baseline,
            fit=fit,
            calibration_id=args.id,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote calibration: {output}")
    print(
        "Selected weights:",
        ",".join(f"{config['weights'][expert]:.2f}" for expert in ALL_EXPERTS),
        f"heuristic_weight={config['heuristic_weight']:.2f}",
        f"threshold={config['threshold']:.6f}",
        f"heldout_margin={config['metrics']['heldout']['margin']:.6f}",
    )


if __name__ == "__main__":
    main()
