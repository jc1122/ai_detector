"""Unit tests for detector_daemon JSONL daemon behavior."""

from __future__ import annotations

import argparse
import io
import json
import unittest
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_ensemble
import detector_daemon


class _FakeTorch:
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    def __init__(self) -> None:
        self.device_calls: list[str] = []
        self.set_threads_calls: list[int] = []

    @staticmethod
    def tensor(*_args: object, **_kwargs: object) -> object:
        return object()

    @staticmethod
    def long() -> object:
        return object()

    @staticmethod
    def device(value: str) -> str:
        return value

    @staticmethod
    def set_num_threads(value: int) -> None:
        return value

    @staticmethod
    def in_fallback_mode() -> bool:
        return False

    cuda = _DummyCuda()


class _FakeTorchForScoring(_FakeTorch):
    @staticmethod
    def get_num_threads() -> int:
        return 4


class _FakeModel:
    def __init__(self) -> None:
        self.to_calls: list[object] = []

    def to(self, device: object) -> "_FakeModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, *args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=None)

    def cpu(self) -> "_FakeModel":
        return self


def _parse_args(*args: str) -> argparse.Namespace:
    return detector_daemon.parse_args(list(args))


class WeightsParsingTests(unittest.TestCase):
    def _normalize(self, request_weights: object, *, defaults: dict[str, float] | None = None) -> dict[str, float]:
        return detector_daemon._normalize_weights_from_request(
            request_weights,
            defaults or {"meld": 0.34, "tmr": 0.33, "raid": 0.33},
            ("meld", "tmr", "raid"),
        )

    def test_parse_weights_from_string_and_list(self) -> None:
        self.assertEqual(
            self._normalize("1,2,1"),
            {"meld": 0.25, "tmr": 0.5, "raid": 0.25},
        )
        self.assertEqual(
            self._normalize([1, 2]),
            {"meld": 1 / 3, "tmr": 2 / 3, "raid": 0.0},
        )

    def test_parse_string_weights_reject_non_finite(self) -> None:
        for value in ("nan,1,0", "inf,1,0", "-inf,1,0"):
            with self.assertRaises(RuntimeError):
                self._normalize(value)

    def test_parse_weights_reject_non_finite_list_or_dict(self) -> None:
        with self.assertRaises(RuntimeError):
            self._normalize([float("nan"), 1, 0])
        with self.assertRaises(RuntimeError):
            self._normalize({"meld": float("inf"), "tmr": 1, "raid": 1})

    def test_parse_dict_weights_is_validated_and_normalized(self) -> None:
        self.assertEqual(
            self._normalize({"meld": 2, "tmr": 1, "raid": 1}),
            {"meld": 0.5, "tmr": 0.25, "raid": 0.25},
        )

        with self.assertRaises(RuntimeError):
            self._normalize({"meld": 2, "tmr": 1})

        with self.assertRaises(RuntimeError):
            self._normalize({"meld": -1, "tmr": 1, "raid": 0})

        with self.assertRaises(RuntimeError):
            self._normalize({"meld": 0, "tmr": 0, "raid": 0})

    def test_parse_dict_weights_reject_negative_value_with_positive_total(self) -> None:
        with self.assertRaises(RuntimeError):
            self._normalize({"meld": -1, "tmr": 2, "raid": 2})


class DaemonScoringTests(unittest.TestCase):
    def _build_daemon(self, *extra_argv: str) -> detector_daemon.DetectorDaemon:
        args = _parse_args(*extra_argv)
        return detector_daemon.DetectorDaemon(args)

    def _assert_scoring_contract(self, payload: dict[str, object]) -> None:
        for key in ("text_preview", "weights", "experts", "ensemble", "calibration", "device"):
            self.assertIn(key, payload)

        self.assertIsInstance(payload["weights"], dict)
        self.assertAlmostEqual(sum(payload["weights"].values()), 1.0, delta=1e-6)
        for value in payload["weights"].values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        experts = payload["experts"]
        self.assertTrue({"meld", "tmr", "raid"}.issubset(set(experts)))
        for expert_payload in experts.values():
            for key in ("ai_score", "human_score", "ai_probability", "human_probability", "chunks", "loaded"):
                self.assertIn(key, expert_payload)
            if expert_payload["loaded"]:
                self.assertNotIn("notes", expert_payload)
                for key in ("ai_score", "human_score", "ai_probability", "human_probability"):
                    value = expert_payload[key]
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                self.assertAlmostEqual(
                    expert_payload["human_probability"],
                    1.0 - expert_payload["ai_probability"],
                    delta=1e-6,
                )
            else:
                self.assertIn("notes", expert_payload)
                self.assertIsNone(expert_payload["ai_score"])
                self.assertIsNone(expert_payload["human_score"])

        ensemble = payload["ensemble"]
        for key in ("ai_score", "human_score", "ai_probability", "human_probability", "threshold", "label"):
            self.assertIn(key, ensemble)

        for key in ("ai_score", "human_score", "ai_probability", "human_probability"):
            value = ensemble[key]
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(
            ensemble["human_probability"],
            1.0 - ensemble["ai_probability"],
            delta=1e-6,
        )

        json.dumps(payload, allow_nan=False)

        calibration = payload["calibration"]
        for key in ("status", "calibrated", "message"):
            self.assertIn(key, calibration)

    def test_parse_args_rejects_non_finite_weights(self) -> None:
        for value in ("nan,1,0", "inf,1,0", "-inf,1,0"):
            with self.assertRaises(SystemExit):
                with patch("sys.stderr", new=io.StringIO()):
                    _parse_args("--weights", value)

    def test_health_reports_loaded_experts_threads_and_device(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_meld, patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_tmr, patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_raid:
            daemon = self._build_daemon("--experts", "meld,tmr", "--threads", "4")
            health = daemon._handle_request({"command": "health"})

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["command"], "health")
        self.assertCountEqual(health["loaded_experts"], ["meld", "tmr"])
        self.assertEqual(health["threads"], 4)
        self.assertEqual(health["device"], "cpu")
        load_meld.assert_called_once()
        load_tmr.assert_called_once()
        load_raid.assert_not_called()

    def test_scoring_reuses_preloaded_models(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        score = iter(
            [
                run_ensemble.ExpertResult(ai_probability=0.1, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.2, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.3, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.5, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.6, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.7, chunks=1),
            ]
        )
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_meld, patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_tmr, patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_raid, patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=lambda *args, **kwargs: next(score),
        ):
            daemon = self._build_daemon()
            first = daemon._handle_request({"text": "first", "weights": "1,1,1"})
            second = daemon._handle_request({"text": "second", "weights": [1, 1, 1]})

        load_meld.assert_called_once()
        load_tmr.assert_called_once()
        load_raid.assert_called_once()
        self._assert_scoring_contract(first)
        self._assert_scoring_contract(second)

    def test_default_weights_use_preloaded_expert_subset(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        score = iter([run_ensemble.ExpertResult(ai_probability=0.2, chunks=1)])
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_meld, patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_tmr, patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_raid, patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=lambda *args, **kwargs: next(score),
        ):
            daemon = self._build_daemon("--profile", "legacy-default", "--experts", "tmr")
            response = daemon._handle_request({"text": "hello"})

        load_tmr.assert_called_once()
        load_meld.assert_not_called()
        load_raid.assert_not_called()
        self._assert_scoring_contract(response)
        self.assertEqual(response["weights"], {"meld": 0.0, "tmr": 1.0, "raid": 0.0})
        self.assertFalse(response["experts"]["meld"]["loaded"])
        self.assertTrue(response["experts"]["tmr"]["loaded"])
        self.assertFalse(response["experts"]["raid"]["loaded"])

    def test_explicit_request_raises_for_unloaded_positive_expert(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ):
            daemon = self._build_daemon("--experts", "meld")
            response = daemon._handle_line(
                '{"text":"hello","weights":{"meld":0.0,"tmr":1.0,"raid":0.0}}'
            )

        self.assertIn("error", response)
        self.assertIn("non-preloaded", response["error"])

    def test_explicit_request_rejects_negative_dict_weights_with_positive_total(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ):
            daemon = self._build_daemon()
            response = daemon._handle_line('{"text":"hello","weights":{"meld":-1,"tmr":2,"raid":2}}')

        self.assertIn("error", response)
        self.assertIn("non-negative", response["error"])

    def test_build_payload_rejects_negative_normalized_weights(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=[
                run_ensemble.ExpertResult(ai_probability=0.4, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.6, chunks=1),
            ],
        ):
            daemon = self._build_daemon("--experts", "tmr,raid")
            with self.assertRaises(RuntimeError):
                daemon._build_payload(
                    text="hello",
                    weights={"meld": -0.25, "tmr": 0.5, "raid": 0.75},
                    threshold=0.5,
                    overlap=128,
                    batch_size=1,
                    max_chunks=None,
                    quiet=False,
                    heuristic_weight=0.0,
                    calibration_applied_to_weights=False,
                    calibration_applied_to_threshold=False,
                    calibration_applied_to_heuristic_weight=False,
                )

    def test_quiet_request_strings_are_parsed_explicitly(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        score = iter(
            [
                run_ensemble.ExpertResult(ai_probability=0.2, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.3, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.4, chunks=1),
            ]
        )
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_meld, patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_tmr, patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ) as load_raid, patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=lambda *args, **kwargs: next(score),
        ), patch.object(detector_daemon.run_ensemble, "_suppress_stderr_fd") as suppress:
            daemon = self._build_daemon("--experts", "meld")
            daemon._handle_request({"text": "hello", "weights": [1, 0], "quiet": "false"})
            daemon._handle_request({"text": "hello", "weights": [1, 0], "quiet": "true"})
            daemon._handle_request({"text": "hello", "weights": [1, 0], "quiet": False})

        load_meld.assert_called_once()
        load_tmr.assert_not_called()
        load_raid.assert_not_called()
        suppress.assert_called_once()

    def test_serve_outputs_scoring_contract_jsonl(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=[
                run_ensemble.ExpertResult(ai_probability=0.1, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.4, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.2, chunks=1),
            ],
        ):
            daemon = self._build_daemon()
            input_stream = io.StringIO(
                '{"text":"To jest dłuższy tekst testowy dla lokalnego sprawdzenia profilu kalibracji."}\n'
            )
            output_stream = io.StringIO()
            daemon.serve(stream_in=input_stream, stream_out=output_stream)

        lines = output_stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self._assert_scoring_contract(response)
        self.assertAlmostEqual(response["weights"]["meld"], 0.3)
        self.assertAlmostEqual(response["weights"]["tmr"], 0.0)
        self.assertAlmostEqual(response["weights"]["raid"], 0.1)
        self.assertAlmostEqual(response["weights"]["heuristic"], 0.6)
        self.assertEqual(response["calibration"]["status"], "operating_point_calibrated")

    def test_short_default_profile_request_marks_heuristic_calibration_unapplied(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(
            detector_daemon.run_ensemble,
            "_score_with_model",
            side_effect=[
                run_ensemble.ExpertResult(ai_probability=0.2, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.4, chunks=1),
            ],
        ):
            daemon = self._build_daemon()
            response = daemon._handle_request({"text": "hello"})

        self._assert_scoring_contract(response)
        self.assertEqual(response["ensemble"]["heuristic_weight"], 0.0)
        self.assertEqual(response["weights"], {"meld": 0.75, "tmr": 0.0, "raid": 0.25})
        self.assertFalse(response["experts"]["heuristic"]["loaded"])
        self.assertTrue(response["calibration"]["applied_to_weights"])
        self.assertTrue(response["calibration"]["applied_to_threshold"])
        self.assertFalse(response["calibration"]["applied_to_heuristic_weight"])

    def test_serve_rejects_non_finite_numeric_outputs(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(daemon := detector_daemon.DetectorDaemon(_parse_args()), "_handle_request", return_value={
            "ensemble": {"ai_probability": float("nan")},
        }):
            input_stream = io.StringIO('{"text":"hello"}\n')
            output_stream = io.StringIO()
            daemon.serve(stream_in=input_stream, stream_out=output_stream)

        lines = output_stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertIn("error", response)
        self.assertIn("non-finite", response["error"])
        self.assertNotIn("NaN", lines[0])
        self.assertNotIn("Infinity", lines[0])


    def test_shutdown_unloads_models_and_collects_garbage(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ), patch("gc.collect") as collect:
            daemon = self._build_daemon()
            response = daemon._handle_request({"command": "shutdown"})
            shutdown_again = daemon._handle_line("{\"command\": \"health\"}")

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["command"], "shutdown")
        self.assertTrue(response["ack"])
        self.assertEqual(response["loaded_experts"], [])
        self.assertEqual(shutdown_again["status"], "ok")
        self.assertEqual(shutdown_again["loaded_experts"], [])
        self.assertTrue(collect.called)
        self.assertEqual(len(daemon.loaded_experts), 0)

    def test_bad_request_returns_error_json(self) -> None:
        torch = _FakeTorchForScoring()
        fake_model = _FakeModel()
        fake_tokenizer = MagicMock()
        with patch.object(detector_daemon.run_ensemble, "_ensure_torch", return_value=torch), patch.object(
            detector_daemon.run_ensemble, "load_meld", return_value=(fake_model, fake_tokenizer, 128)
        ), patch.object(detector_daemon.run_ensemble, "load_tmr", return_value=(fake_model, fake_tokenizer, 128)), patch.object(
            detector_daemon.run_ensemble, "load_raid", return_value=(fake_model, fake_tokenizer, 128)
        ):
            daemon = self._build_daemon("--experts", "meld")
            bad_json = daemon._handle_line("{not valid json")
            bad_payload = daemon._handle_line("{}")
        self.assertIn("error", bad_json)
        self.assertIn("error", bad_payload)
        self.assertIn("string `text` field", bad_payload["error"])
