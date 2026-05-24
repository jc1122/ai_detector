"""Tests for operating-point calibration artifacts and CLI wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import calibrate_detector
import calibration_config
import detector_daemon
import run_ensemble


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BROAD_RESULT = PROJECT_ROOT / "data/evaluation/polish_pre2020_technical_papers/broad_eval_2026-05-24.json"
WINDOW_RESULT = PROJECT_ROOT / "data/evaluation/polish_pre2020_technical_papers/broad_eval_windows_2026-05-24.json"
CALIBRATION_FILE = PROJECT_ROOT / "data/evaluation/polish_pre2020_technical_papers/calibration_pl_ood_2026-05-24.json"
SOURCE_CACHE_MANIFEST = PROJECT_ROOT / "data/evaluation/polish_pre2020_technical_papers/sources/manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibrationConfigTests(unittest.TestCase):
    def test_load_calibration_config_normalizes_weights(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = Path(workspace) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "test_calibration",
                        "weights": {"meld": 8, "tmr": 0, "raid": 2},
                        "threshold": 0.49,
                        "method": "test",
                    }
                ),
                encoding="utf-8",
            )

            loaded = calibration_config.load_calibration_config(path)

        self.assertEqual(loaded.calibration_id, "test_calibration")
        self.assertEqual(loaded.weights, {"meld": 0.8, "tmr": 0.0, "raid": 0.2})
        self.assertEqual(loaded.threshold, 0.49)
        self.assertEqual(loaded.heuristic_weight, 0.0)

    def test_load_calibration_config_rejects_bad_schema(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = Path(workspace) / "calibration.json"
            path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                calibration_config.load_calibration_config(path)

    def test_payload_marks_applied_operating_point(self) -> None:
        config = calibration_config.CalibrationConfig(
            source_path=Path("calibration.json"),
            schema_version=1,
            calibration_id="test",
            weights={"meld": 1.0, "tmr": 0.0, "raid": 0.0},
            threshold=0.5,
            heuristic_weight=0.0,
            method="test",
            description="",
            payload={},
        )

        payload = calibration_config.calibration_payload(
            config,
            applied_to_weights=True,
            applied_to_threshold=False,
        )

        self.assertEqual(payload["status"], "operating_point_calibrated")
        self.assertTrue(payload["calibrated"])
        self.assertFalse(payload["probability_calibrated"])
        self.assertTrue(payload["applied_to_weights"])
        self.assertFalse(payload["applied_to_threshold"])


class CalibrationFittingTests(unittest.TestCase):
    def test_source_paper_cache_manifest_points_to_committed_files(self) -> None:
        manifest = json.loads(SOURCE_CACHE_MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["items"]), 12)
        for item in manifest["items"]:
            pdf = PROJECT_ROOT / item["pdf_path"]
            clean_text = PROJECT_ROOT / item["clean_text_path"]
            self.assertTrue(pdf.is_file(), pdf)
            self.assertTrue(clean_text.is_file(), clean_text)
            self.assertEqual(pdf.stat().st_size, item["pdf_size_bytes"])
            self.assertEqual(clean_text.stat().st_size, item["clean_text_size_bytes"])
            self.assertEqual(_sha256(pdf), item["pdf_sha256"])
            self.assertEqual(_sha256(clean_text), item["clean_text_sha256"])

    def test_committed_runtime_calibration_matches_builtin_profile(self) -> None:
        artifact = calibration_config.load_calibration_config(CALIBRATION_FILE)
        profile = calibration_config.load_builtin_profile("pl-technical-ood")

        self.assertEqual(artifact.calibration_id, profile.calibration_id)
        self.assertEqual(artifact.weights, profile.weights)
        self.assertEqual(artifact.threshold, profile.threshold)
        self.assertEqual(artifact.heuristic_weight, profile.heuristic_weight)
        self.assertEqual(artifact.method, profile.method)
        self.assertEqual(artifact.payload["source_result_file"], str(WINDOW_RESULT.relative_to(PROJECT_ROOT)))

    def test_window_artifact_profile_separates_controls(self) -> None:
        records = json.loads(WINDOW_RESULT.read_text(encoding="utf-8"))["records"]
        artifact = calibration_config.load_calibration_config(CALIBRATION_FILE)

        def score(record: dict[str, object]) -> float:
            model = sum(artifact.weights[name] * record["scores"][name] for name in ("meld", "tmr", "raid"))
            return (1.0 - artifact.heuristic_weight) * model + artifact.heuristic_weight * record["scores"]["heuristic"]

        human = [score(record) for record in records if record["label"] == "human_pre2020"]
        ai = [score(record) for record in records if record["label"] == "synthetic_ai_control"]

        self.assertEqual(sum(value >= artifact.threshold for value in human), 0)
        self.assertEqual(sum(value < artifact.threshold for value in ai), 0)
        self.assertGreater(min(ai) - max(human), 0.05)

    def test_window_result_regenerates_committed_hybrid_calibration(self) -> None:
        baseline = json.loads(WINDOW_RESULT.read_text(encoding="utf-8"))

        fit = calibrate_detector.fit_calibration(
            baseline,
            grid_step=0.05,
            view="all",
            human_quality="pl_clean",
            test_fraction=0.34,
            split_seed="pl-technical-ood-v2",
            fixed_weights={"meld": 0.75, "tmr": 0.0, "raid": 0.25},
            fit_heuristic_weight=True,
            heuristic_grid_step=0.05,
            max_heuristic_weight=0.60,
        )
        config = calibrate_detector.build_config(
            baseline_path=WINDOW_RESULT.relative_to(PROJECT_ROOT),
            baseline=baseline,
            fit=fit,
            calibration_id="pl_technical_ood_2026_05_24",
        )
        committed = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))

        self.assertEqual(config["weights"], committed["weights"])
        self.assertEqual(config["heuristic_weight"], committed["heuristic_weight"])
        self.assertEqual(config["threshold"], committed["threshold"])
        self.assertEqual(config["method"], committed["method"])
        self.assertEqual(config["metrics"]["heldout"]["false_positives_at_threshold"], 0)
        self.assertEqual(config["metrics"]["heldout"]["false_negatives_at_threshold"], 0)

    def test_broad_result_fit_selects_separating_weights(self) -> None:
        baseline = json.loads(BROAD_RESULT.read_text(encoding="utf-8"))

        fit = calibrate_detector.fit_calibration(
            baseline,
            grid_step=0.05,
            view="all",
            human_quality="pl_clean",
            test_fraction=0.34,
            split_seed="pl-technical-ood-v1",
        )
        best = fit["best"]

        self.assertEqual(best["false_positives"], 0)
        self.assertEqual(best["false_negatives"], 0)
        self.assertGreater(best["margin"], 0.09)
        self.assertEqual(best["weights"], {"meld": 0.75, "tmr": 0.0, "raid": 0.25})
        self.assertGreater(best["threshold"], best["human"]["max"])
        self.assertLess(best["threshold"], best["ai_control"]["min"])
        self.assertEqual(fit["metrics"]["heldout"]["false_positives_at_threshold"], 0)
        self.assertEqual(fit["metrics"]["heldout"]["false_negatives_at_threshold"], 0)

    def test_build_config_matches_runtime_schema(self) -> None:
        baseline = json.loads(BROAD_RESULT.read_text(encoding="utf-8"))
        fit = calibrate_detector.fit_calibration(
            baseline,
            grid_step=0.05,
            view="all",
            human_quality="pl_clean",
            test_fraction=0.34,
            split_seed="pl-technical-ood-v1",
        )

        with tempfile.TemporaryDirectory() as workspace:
            path = Path(workspace) / "calibration.json"
            config = calibrate_detector.build_config(
                baseline_path=BROAD_RESULT,
                baseline=baseline,
                fit=fit,
                calibration_id="test_pl",
            )
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = calibration_config.load_calibration_config(path)

        self.assertEqual(loaded.calibration_id, "test_pl")
        self.assertEqual(loaded.weights, {"meld": 0.75, "tmr": 0.0, "raid": 0.25})


class CalibrationWiringTests(unittest.TestCase):
    def _write_config(self, workspace: str) -> Path:
        path = Path(workspace) / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "test_runtime_calibration",
                    "weights": {"meld": 0.75, "tmr": 0.0, "raid": 0.25},
                    "threshold": 0.51,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_run_ensemble_parse_args_applies_calibration_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = self._write_config(workspace)
            args = run_ensemble.parse_args(["--calibration-file", str(path), "--text", "hello"])

        self.assertEqual(args.weights, {"meld": 0.75, "tmr": 0.0, "raid": 0.25})
        self.assertEqual(args.threshold, 0.51)
        self.assertEqual(args.heuristic_weight, 0.0)
        self.assertTrue(args.calibration_applied_to_weights)
        self.assertTrue(args.calibration_applied_to_threshold)

    def test_run_ensemble_default_profile_is_packaged_pl_calibration(self) -> None:
        args = run_ensemble.parse_args(["--text", "hello"])

        self.assertEqual(args.weights, {"meld": 0.75, "tmr": 0.0, "raid": 0.25})
        self.assertAlmostEqual(args.threshold, 0.4345518078804016)
        self.assertAlmostEqual(args.heuristic_weight, 0.6)
        self.assertEqual(args.calibration.calibration_id, "pl_technical_ood_2026_05_24")

    def test_run_ensemble_explicit_weights_override_calibration_weights(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = self._write_config(workspace)
            args = run_ensemble.parse_args(
                [
                    "--calibration-file",
                    str(path),
                    "--weights",
                    "1,0,0",
                    "--threshold",
                    "0.6",
                    "--text",
                    "hello",
                ]
            )

        self.assertEqual(args.weights, {"meld": 1.0, "tmr": 0.0, "raid": 0.0})
        self.assertEqual(args.threshold, 0.6)
        self.assertEqual(args.heuristic_weight, 0.0)
        self.assertFalse(args.calibration_applied_to_weights)
        self.assertFalse(args.calibration_applied_to_threshold)

    def test_daemon_parse_args_applies_calibration_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = self._write_config(workspace)
            args = detector_daemon.parse_args(["--calibration-file", str(path)])

        self.assertEqual(args.weights, {"meld": 0.75, "tmr": 0.0, "raid": 0.25})
        self.assertEqual(args.threshold, 0.51)
        self.assertEqual(args.heuristic_weight, 0.0)
        self.assertTrue(args.calibration_applied_to_weights)
        self.assertTrue(args.calibration_applied_to_threshold)


if __name__ == "__main__":
    unittest.main()
