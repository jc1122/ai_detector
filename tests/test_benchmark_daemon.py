"""Tests for benchmark helper utility behavior."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
import unittest


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_daemon.py"
    spec = importlib.util.spec_from_file_location("benchmark_daemon", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_daemon"] = module
    spec.loader.exec_module(module)
    return module


benchmark_daemon = _load_benchmark_module()


class BenchmarkDaemonTests(unittest.TestCase):
    def test_percentile_empty_list_returns_nan(self) -> None:
        self.assertTrue(math.isnan(benchmark_daemon._percentile([], p=0.5)))

    def test_percentile_returns_bounds_for_singleton(self) -> None:
        self.assertEqual(benchmark_daemon._percentile([7.5], p=0.5), 7.5)

    def test_percentile_interpolates_between_values(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]

        self.assertEqual(benchmark_daemon._percentile(values, p=0.0), 10.0)
        self.assertEqual(benchmark_daemon._percentile(values, p=1.0), 40.0)
        self.assertAlmostEqual(benchmark_daemon._percentile(values, p=0.5), 25.0)

    def test_build_request_payload(self) -> None:
        payload = benchmark_daemon._build_request_payload(
            text="hello",
            batch_size=4,
            max_chunks=2,
            quiet=True,
        )

        self.assertEqual(
            payload,
            {
                "text": "hello",
                "batch_size": 4,
                "max_chunks": 2,
                "quiet": True,
            },
        )

    def test_validate_scoring_response_checks_required_keys(self) -> None:
        valid_payload = {
            "text_preview": "hello",
            "weights": {"meld": 0.2, "tmr": 0.4, "raid": 0.4},
            "experts": {
                "meld": {
                    "ai_score": 0.1,
                    "human_score": 0.9,
                    "ai_probability": 0.1,
                    "human_probability": 0.9,
                    "chunks": 1,
                    "loaded": True,
                },
                "tmr": {
                    "ai_score": 0.2,
                    "human_score": 0.8,
                    "ai_probability": 0.2,
                    "human_probability": 0.8,
                    "chunks": 1,
                    "loaded": True,
                },
                "raid": {
                    "ai_score": 0.3,
                    "human_score": 0.7,
                    "ai_probability": 0.3,
                    "human_probability": 0.7,
                    "chunks": 1,
                    "loaded": True,
                },
            },
            "ensemble": {
                "ai_score": 0.2,
                "human_score": 0.8,
                "ai_probability": 0.2,
                "human_probability": 0.8,
                "threshold": 0.5,
                "label": "human",
            },
            "calibration": {"status": "uncalibrated", "calibrated": False, "message": "ok"},
            "device": "cpu",
        }

        # Should not raise
        benchmark_daemon._validate_scoring_response(valid_payload)

        invalid_payload = dict(valid_payload)
        invalid_payload["ensemble"] = {}
        with self.assertRaises(RuntimeError):
            benchmark_daemon._validate_scoring_response(invalid_payload)


if __name__ == "__main__":
    unittest.main()
