"""Unit tests for CLI-like behavior of run_ensemble without loading checkpoints."""

from __future__ import annotations

import argparse
import io
import json
import os
import types
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_ensemble


class DummyTorch:
    class _DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    def __init__(self) -> None:
        self.cuda = self._DummyCuda()

    @staticmethod
    def device(value: str) -> str:
        return value


class _FakeLogits:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def float(self) -> "_FakeLogits":
        return self


class _FakeSoftmaxOutput:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def __getitem__(self, item):
        row_selector, col = item
        if col != 1:
            raise IndexError("Only second column is supported in this fake softmax.")
        if isinstance(row_selector, slice):
            if row_selector != slice(None, None, None):
                raise IndexError("Only full row slices are supported in this fake softmax.")
            rows = self._values[row_selector]
        else:
            rows = self._values[row_selector : row_selector + 1]
        return [_FakeScalar(row[1]) for row in rows]


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeTorchForScoring:
    long = "long"

    class _InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def inference_mode(self) -> "_FakeTorchForScoring._InferenceMode":
        return self._InferenceMode()

    @staticmethod
    def tensor(data: list[list[int]], dtype: object, device: str | None = None) -> list[list[int]]:
        return data

    @staticmethod
    def softmax(logits: _FakeLogits, dim: int) -> list[_FakeScalar]:
        return _FakeSoftmaxOutput(logits.values)


class _FakeTorchForPadding:
    long = "long"

    class _InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def __init__(self) -> None:
        self.tensor_calls: list[list[list[int]]] = []

    def inference_mode(self) -> "_FakeTorchForPadding._InferenceMode":
        return self._InferenceMode()

    def tensor(self, data: list[list[int]], dtype: object, device: str | None = None) -> list[list[int]]:
        lengths = [len(row) for row in data]
        if lengths and len(set(lengths)) != 1:
            raise AssertionError(f"Ragged tensor passed to torch.tensor: {lengths}")
        self.tensor_calls.append(data)
        return data

    @staticmethod
    def softmax(logits: _FakeLogits, dim: int) -> list[_FakeScalar]:
        return _FakeSoftmaxOutput(logits.values)


class _ArgparseUtils:
    def _parse_cli(self, argv: list[str]) -> argparse.Namespace:
        original_argv = sys.argv
        try:
            sys.argv = ["run_ensemble.py", *argv]
            return run_ensemble.parse_args()
        finally:
            sys.argv = original_argv


class ParseWeightsTests(unittest.TestCase, _ArgparseUtils):
    def test_parse_local_files_only(self) -> None:
        self.assertFalse(self._parse_cli([]).local_files_only)
        args = self._parse_cli(["--local-files-only"])
        self.assertTrue(args.local_files_only)

    def test_parse_weights_accepts_two_values_and_adds_zero_for_raid(self) -> None:
        weights = run_ensemble._parse_weights("0.7,0.3")
        self.assertEqual(weights, {"meld": 0.7, "tmr": 0.3, "raid": 0.0})

    def test_parse_weights_accepts_three_values(self) -> None:
        weights = run_ensemble._parse_weights("0.5,0.25,0.25")
        self.assertEqual(weights, {"meld": 0.5, "tmr": 0.25, "raid": 0.25})

    def test_parse_weights_reject_invalid_count(self) -> None:
        with self.assertRaises(SystemExit):
            self._parse_cli(["--weights", "0.5"])

    def test_parse_weights_reject_negative(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run_ensemble._parse_weights("-1,0.5,0.5")

    def test_parse_weights_reject_zero_sum(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            run_ensemble._parse_weights("0,0,0")

    def test_parse_quiet_defaults_to_false(self) -> None:
        args = self._parse_cli([])
        self.assertFalse(args.quiet)

    def test_parse_quiet_flag_enabled(self) -> None:
        args = self._parse_cli(["--quiet"])
        self.assertTrue(args.quiet)


class RunEnsembleTests(unittest.TestCase, _ArgparseUtils):
    def _make_args(self, **kwargs: object) -> argparse.Namespace:
        args = {
            "meld_dir": "meld_model",
            "tmr_dir": "tmr_model",
            "raid_dir": "raid_model",
            "weights": {"meld": 0.34, "tmr": 0.33, "raid": 0.33},
            "threshold": 0.5,
            "overlap": 0,
            "batch_size": 8,
            "max_chunks": None,
            "device": "cpu",
            "local_files_only": False,
        }
        args.update(kwargs)
        return argparse.Namespace(**args)

    def _mock_model_call(self, score_by_expert: list[run_ensemble.ExpertResult] | None = None):
        if score_by_expert is None:
            score_by_expert = [
                run_ensemble.ExpertResult(ai_probability=0.2, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.7, chunks=1),
                run_ensemble.ExpertResult(ai_probability=0.4, chunks=1),
            ]
        scores = iter(score_by_expert)
        return lambda *args, **kwargs: next(scores)

    def test_missing_text_file_reports_runtime_error(self) -> None:
        args = argparse.Namespace(text=None, text_file="missing_text_file.txt")
        with self.assertRaises(RuntimeError) as exc_info:
            run_ensemble._read_text(args)
        self.assertIn("Cannot read --text-file", str(exc_info.exception))

    def test_main_reports_missing_text_file_without_trace(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            missing = Path(workspace) / "missing_input.txt"
            with patch("sys.argv", ["run_ensemble.py", "--text-file", str(missing)]), patch(
                "sys.stderr", new=io.StringIO()
            ) as stderr:
                with self.assertRaises(SystemExit):
                    run_ensemble.main()
                error_text = stderr.getvalue()

        self.assertIn("Cannot read --text-file", error_text)
        self.assertNotIn("Traceback", error_text)

    def test_meld_backbone_import_error(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        with tempfile.TemporaryDirectory() as meld_dir:
            with patch.dict("sys.modules", {"transformers": fake_transformers}):
                with self.assertRaises(RuntimeError) as exc_info:
                    run_ensemble._load_meld_backbone(
                        "hf_backbone", Path(meld_dir), local_files_only=False
                    )
                self.assertIn(
                    "transformers is required to load the MELD backbone.",
                    str(exc_info.exception),
                )

    def test_main_reports_loader_dependency_error(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "load_meld", side_effect=ImportError("No module named 'safetensors'")
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid, patch("sys.argv", ["run_ensemble.py", "--text", "hello", "--weights", "1,0,0"]), patch(
            "sys.stderr", new=io.StringIO()
        ) as stderr:
            with self.assertRaises(SystemExit):
                run_ensemble.main()
            error_text = stderr.getvalue()

        self.assertIn(
            "Failed to initialize model for expert 'meld' due to a missing dependency",
            error_text,
        )
        self.assertNotIn("Traceback", error_text)
        load_meld.assert_called_once()
        load_tmr.assert_not_called()
        load_raid.assert_not_called()

    def test_main_reports_missing_model_directory_without_traceback(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "load_meld", side_effect=OSError("No model directory found: meld_model/config.json")
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid, patch("sys.argv", ["run_ensemble.py", "--text", "hello", "--weights", "1,0,0"]), patch(
            "sys.stderr", new=io.StringIO()
        ) as stderr:
            with self.assertRaises(SystemExit):
                run_ensemble.main()
            error_text = stderr.getvalue()

        self.assertIn(
            "Failed to initialize model for expert 'meld' from 'meld_model':",
            error_text,
        )
        self.assertNotIn("Traceback", error_text)
        load_meld.assert_called_once()
        load_tmr.assert_not_called()
        load_raid.assert_not_called()

    def test_run_ensemble_wraps_oserror_from_loader(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble,
            "load_tmr",
            side_effect=OSError("Failed to load weights from transformers hub config file"),
        ) as load_tmr, patch.object(run_ensemble, "load_meld") as load_meld, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid:
            with self.assertRaises(RuntimeError) as exc_info:
                run_ensemble.run_ensemble(
                    "hello",
                    self._make_args(weights={"meld": 0.0, "tmr": 1.0, "raid": 0.0}),
                )

        self.assertIn(
            "Failed to initialize model for expert 'tmr' from 'tmr_model':",
            str(exc_info.exception),
        )
        load_tmr.assert_called_once()
        load_meld.assert_not_called()
        load_raid.assert_not_called()

    def test_output_contains_ai_and_human_probabilities_and_scores(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "load_meld"
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid, patch.object(
            run_ensemble, "_score_with_model", side_effect=self._mock_model_call()
        ):
            fake_model = MagicMock()
            fake_tokenizer = MagicMock()
            load_meld.return_value = (fake_model, fake_tokenizer, 512)
            load_tmr.return_value = (fake_model, fake_tokenizer, 512)
            load_raid.return_value = (fake_model, fake_tokenizer, 512)

            result = run_ensemble.run_ensemble("The sky is clear.", self._make_args())

            self.assertIn("ensemble", result)
            self.assertIn("experts", result)
            ensemble = result["ensemble"]
            for key in ("ai_score", "human_score", "ai_probability", "human_probability"):
                self.assertIn(key, ensemble)
            self.assertAlmostEqual(ensemble["human_probability"], 1.0 - ensemble["ai_probability"])

            for expert in ("meld", "tmr", "raid"):
                self.assertIn(expert, result["experts"])
                payload = result["experts"][expert]
                for key in ("ai_score", "human_score", "ai_probability", "human_probability"):
                    self.assertIn(key, payload)
                if result["weights"][expert] == 0.0:
                    self.assertFalse(payload["loaded"])
                else:
                    self.assertTrue(payload["loaded"])

    def test_model_with_zero_weight_is_not_loaded(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "load_meld"
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid, patch.object(
            run_ensemble, "_score_with_model", side_effect=self._mock_model_call(
                [
                    run_ensemble.ExpertResult(ai_probability=0.42, chunks=1),
                ]
            )
        ):
            fake_model = MagicMock()
            fake_tokenizer = MagicMock()
            load_meld.return_value = (fake_model, fake_tokenizer, 512)
            load_tmr.return_value = (fake_model, fake_tokenizer, 512)
            load_raid.return_value = (fake_model, fake_tokenizer, 512)

            result = run_ensemble.run_ensemble(
                "This is a short sentence.",
                self._make_args(weights={"meld": 1.0, "tmr": 0.0, "raid": 0.0}),
            )

            load_meld.assert_called_once()
            load_tmr.assert_not_called()
            load_raid.assert_not_called()
            self.assertFalse(result["experts"]["tmr"]["loaded"])
            self.assertFalse(result["experts"]["raid"]["loaded"])
            for payload in (result["experts"]["tmr"], result["experts"]["raid"]):
                self.assertIsNone(payload["ai_score"])
                self.assertIsNone(payload["human_score"])
                self.assertIsNone(payload["ai_probability"])
                self.assertIsNone(payload["human_probability"])

    def test_batch_size_and_max_chunks_validation(self) -> None:
        with self.assertRaises(SystemExit):
            self._parse_cli(["--batch-size", "0"])
        with self.assertRaises(SystemExit):
            self._parse_cli(["--batch-size", "-1"])
        with self.assertRaises(SystemExit):
            self._parse_cli(["--max-chunks", "0"])
        with self.assertRaises(SystemExit):
            self._parse_cli(["--max-chunks", "-1"])
        args = self._parse_cli(["--max-chunks", "3"])
        self.assertEqual(args.max_chunks, 3)

    def test_empty_input_returns_error(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "_score_with_model"
        ) as score_with_model:
            with self.assertRaises(RuntimeError) as exc_info:
                run_ensemble.run_ensemble("", self._make_args(weights={"meld": 0.5, "tmr": 0.5, "raid": 0.0}))
            self.assertEqual(str(exc_info.exception), "No input text provided.")
            score_with_model.assert_not_called()

    def test_zero_overlap_passes_stride_zero_to_tokenizer(self) -> None:
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        def run_model(*_args: object, **_kwargs: object) -> _FakeLogits:
            return _FakeLogits([[0.1, 0.6]])

        result = run_ensemble._score_with_model(
            run_model,
            tokenizer,
            text="some short text",
            max_length=5,
            torch_module=_FakeTorchForScoring(),
            device="cpu",
            overlap=0,
            batch_size=4,
            max_chunks=None,
        )
        self.assertEqual(tokenizer.call_args.kwargs["stride"], 0)
        self.assertEqual(result.ai_probability, 0.6)
        self.assertEqual(result.chunks, 1)

    def test_cuda_unavailable_raises_before_loading_models(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()) as ensure_torch, patch.object(
            run_ensemble, "load_meld"
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid:
            with self.assertRaises(RuntimeError) as exc_info:
                run_ensemble.run_ensemble("some text", self._make_args(device="cuda"))
            self.assertIn("CUDA was requested but is not available", str(exc_info.exception))
            ensure_torch.assert_called_once()
            load_meld.assert_not_called()
            load_tmr.assert_not_called()
            load_raid.assert_not_called()

    def test_loaders_receive_local_files_only_flag(self) -> None:
        with patch.object(run_ensemble, "_ensure_torch", return_value=DummyTorch()), patch.object(
            run_ensemble, "load_meld"
        ) as load_meld, patch.object(run_ensemble, "load_tmr") as load_tmr, patch.object(
            run_ensemble, "load_raid"
        ) as load_raid, patch.object(
            run_ensemble, "_score_with_model", side_effect=self._mock_model_call()
        ):
            fake_model = MagicMock()
            fake_tokenizer = MagicMock()
            load_meld.return_value = (fake_model, fake_tokenizer, 512)
            load_tmr.return_value = (fake_model, fake_tokenizer, 512)
            load_raid.return_value = (fake_model, fake_tokenizer, 512)

            run_ensemble.run_ensemble("some text", self._make_args(local_files_only=True))

            load_meld.assert_called_once_with(Path("meld_model"), local_files_only=True)
            load_tmr.assert_called_once_with(Path("tmr_model"), local_files_only=True)
            load_raid.assert_called_once_with(Path("raid_model"), local_files_only=True)

    def test_meld_backbone_loads_from_cfg_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as meld_dir:
            model_dir = Path(meld_dir)

            fake_transformers = SimpleNamespace(
                AutoModel=SimpleNamespace(from_pretrained=MagicMock())
            )
            with patch.dict("sys.modules", {"transformers": fake_transformers}):
                backbone = MagicMock()
                fake_transformers.AutoModel.from_pretrained.return_value = backbone

                loaded = run_ensemble._load_meld_backbone("hf_backbone", model_dir, local_files_only=False)

                self.assertIs(loaded, backbone)
                fake_transformers.AutoModel.from_pretrained.assert_called_once_with(
                    "hf_backbone", local_files_only=False
                )

    def test_meld_backbone_local_only_requires_cached_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as meld_dir:
            model_dir = Path(meld_dir)

            fake_transformers = SimpleNamespace(
                AutoModel=SimpleNamespace(from_pretrained=MagicMock())
            )
            with patch.dict("sys.modules", {"transformers": fake_transformers}):
                fake_transformers.AutoModel.from_pretrained.side_effect = RuntimeError("not in cache")

                with self.assertRaises(RuntimeError) as exc_info:
                    run_ensemble._load_meld_backbone("hf_backbone", model_dir, local_files_only=True)

                self.assertEqual(fake_transformers.AutoModel.from_pretrained.call_count, 1)
                self.assertEqual(
                    fake_transformers.AutoModel.from_pretrained.call_args.args[0],
                    "hf_backbone",
                )
                self.assertTrue(fake_transformers.AutoModel.from_pretrained.call_args.kwargs["local_files_only"])
                message = str(exc_info.exception)
                self.assertIn("local Hugging Face cache", message)

    def test_batch_padding_uses_pad_token_and_zero_attention_masks(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 99
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4], [5, 6]],
            "attention_mask": [[1, 1, 1, 1], [1, 1]],
        }

        fake_torch = _FakeTorchForPadding()

        def run_model(*_args: object, **_kwargs: object) -> _FakeLogits:
            return _FakeLogits([[0.1, 0.4], [0.2, 0.8]])

        result = run_ensemble._score_with_model(
            run_model,
            tokenizer,
            text="text with variable chunks",
            max_length=8,
            torch_module=fake_torch,
            device="cpu",
            overlap=0,
            batch_size=2,
            max_chunks=None,
        )

        self.assertEqual(len(fake_torch.tensor_calls), 2)
        input_tensor, attention_tensor = fake_torch.tensor_calls
        self.assertEqual(len(set(len(row) for row in input_tensor)), 1)
        self.assertEqual(len(set(len(row) for row in attention_tensor)), 1)
        self.assertEqual(input_tensor[1][2:], [99, 99])
        self.assertEqual(attention_tensor[1][2:], [0, 0])
        self.assertAlmostEqual(result.ai_probability, 0.6)
        self.assertEqual(result.chunks, 2)

    def test_main_plain_output_shows_skipped_for_zero_weight_experts(self) -> None:
        fake_result = {
            "text_preview": "x",
            "weights": {"meld": 0.0, "tmr": 1.0, "raid": 0.0},
            "experts": {
                "meld": {
                    "ai_score": None,
                    "human_score": None,
                    "ai_probability": None,
                    "human_probability": None,
                    "chunks": 0,
                    "loaded": False,
                    "notes": "Not scored because model weight is 0.0.",
                },
                "tmr": {
                    "ai_score": 0.7,
                    "human_score": 0.3,
                    "ai_probability": 0.7,
                    "human_probability": 0.3,
                    "chunks": 1,
                    "loaded": True,
                },
                "raid": {
                    "ai_score": None,
                    "human_score": None,
                    "ai_probability": None,
                    "human_probability": None,
                    "chunks": 0,
                    "loaded": False,
                    "notes": "Not scored because model weight is 0.0.",
                },
            },
            "ensemble": {
                "ai_score": 0.7,
                "human_score": 0.3,
                "ai_probability": 0.7,
                "human_probability": 0.3,
                "threshold": 0.5,
                "label": "ai",
            },
            "calibration": {
                "status": "uncalibrated",
                "calibrated": False,
                "message": "Scores are uncalibrated raw model probabilities. "
                "Provide and use a calibrated model to get calibrated scores.",
            },
            "device": "cpu",
        }

        with patch.object(run_ensemble, "run_ensemble", return_value=fake_result), patch(
            "sys.argv", ["run_ensemble.py", "--weights", "0,1,0", "--text", "x"]
        ), patch("sys.stdout", new=io.StringIO()) as stdout:
            run_ensemble.main()

        output = stdout.getvalue()
        self.assertIn("MELD AI probability: skipped", output)
        self.assertIn("RAID AI probability: skipped", output)
        self.assertIn("Decision: ai", output)

    def test_main_with_quiet_suppresses_inference_stderr(self) -> None:
        fake_result = {
            "text_preview": "hello",
            "weights": {"meld": 0.34, "tmr": 0.33, "raid": 0.33},
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
                    "ai_score": 0.9,
                    "human_score": 0.1,
                    "ai_probability": 0.9,
                    "human_probability": 0.1,
                    "chunks": 1,
                    "loaded": True,
                },
                "raid": {
                    "ai_score": 0.2,
                    "human_score": 0.8,
                    "ai_probability": 0.2,
                    "human_probability": 0.8,
                    "chunks": 1,
                    "loaded": True,
                },
            },
            "ensemble": {
                "ai_score": 0.4,
                "human_score": 0.6,
                "ai_probability": 0.4,
                "human_probability": 0.6,
                "threshold": 0.5,
                "label": "human",
            },
            "calibration": {
                "status": "uncalibrated",
                "calibrated": False,
                "message": "Scores are uncalibrated raw model probabilities.",
            },
            "device": "cpu",
        }

        def run_inference(_text: str, _args: argparse.Namespace) -> dict[str, object]:
            os.write(2, b"noisy third-party chatter")
            return fake_result

        with tempfile.TemporaryFile("w+b") as stderr_capture, patch(
            "run_ensemble.run_ensemble", side_effect=run_inference
        ), patch("sys.argv", ["run_ensemble.py", "--quiet", "--text", "hello"]), patch(
            "sys.stdout", new=io.StringIO()
        ) as stdout:
            original_stderr_fd = os.dup(2)
            os.dup2(stderr_capture.fileno(), 2)
            try:
                run_ensemble.main()
            finally:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)

            stderr_capture.seek(0)
            self.assertEqual(stderr_capture.read().decode("utf-8"), "")

        self.assertIn("Ensemble AI probability:", stdout.getvalue())

    def test_main_quiet_preserves_user_error_to_real_stderr(self) -> None:
        def fail_inference(_text: str, _args: argparse.Namespace) -> dict[str, object]:
            os.write(2, b"noisy third-party chatter")
            raise RuntimeError("failed during inference")

        with tempfile.TemporaryFile("w+b") as stderr_capture, patch(
            "run_ensemble.run_ensemble", side_effect=fail_inference
        ), patch("sys.argv", ["run_ensemble.py", "--quiet", "--text", "hello"]), patch(
            "sys.stdout", new=io.StringIO()
        ) as stdout, patch("sys.stderr", new=io.StringIO()) as stderr:
            original_stderr_fd = os.dup(2)
            os.dup2(stderr_capture.fileno(), 2)
            try:
                with self.assertRaises(SystemExit):
                    run_ensemble.main()
            finally:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)

            stderr_capture.seek(0)
            self.assertEqual(stderr_capture.read().decode("utf-8"), "")

        self.assertIn("Error: failed during inference", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_main_json_output_emits_contract_keys(self) -> None:
        fake_result = {
            "text_preview": "x",
            "weights": {"meld": 0.34, "tmr": 0.33, "raid": 0.33},
            "experts": {
                "meld": {
                    "ai_score": 0.5,
                    "human_score": 0.5,
                    "ai_probability": 0.5,
                    "human_probability": 0.5,
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
                    "ai_score": 0.7,
                    "human_score": 0.3,
                    "ai_probability": 0.7,
                    "human_probability": 0.3,
                    "chunks": 1,
                    "loaded": True,
                },
            },
            "ensemble": {
                "ai_score": 0.48,
                "human_score": 0.52,
                "ai_probability": 0.48,
                "human_probability": 0.52,
                "threshold": 0.5,
                "label": "human",
            },
            "calibration": {
                "status": "uncalibrated",
                "calibrated": False,
                "message": "Scores are uncalibrated raw model probabilities. "
                "Provide and use a calibrated model to get calibrated scores.",
            },
            "device": "cpu",
        }

        with patch.object(run_ensemble, "run_ensemble", return_value=fake_result), patch(
            "sys.argv", ["run_ensemble.py", "--json", "--text", "x"]
        ), patch("sys.stdout", new=io.StringIO()) as stdout:
            run_ensemble.main()

        parsed = json.loads(stdout.getvalue())
        self.assertIn("experts", parsed)
        self.assertIn("ensemble", parsed)
        self.assertIn("calibration", parsed)
        self.assertIn("weights", parsed)
        self.assertIn("device", parsed)
