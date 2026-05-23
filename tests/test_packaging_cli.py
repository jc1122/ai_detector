"""Tests for packaging and CLI entrypoint contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_project_scripts(pyproject_path: Path) -> dict[str, str]:
    """Read only [project.scripts] from pyproject.toml in a version-safe way."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        project = data.get("project", {})
        scripts = project.get("scripts", {})
        return {str(key): str(value) for key, value in dict(scripts).items()}

    return _parse_project_scripts_fallback(pyproject_path.read_text(encoding="utf-8"))


def _remove_toml_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in {"'", '"'}:
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            continue
        if ch == "#" and in_quote is None:
            return line[:idx]
    return line


def _unquote_toml_token(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _parse_project_scripts_fallback(pyproject_text: str) -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts_block = False
    for raw_line in pyproject_text.splitlines():
        line = _remove_toml_comment(raw_line).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            in_scripts_block = section == "project.scripts"
            continue
        if not in_scripts_block or "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        key = _unquote_toml_token(key)
        value = _unquote_toml_token(value)
        scripts[key] = value
    return scripts


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


def _run_import_smoke(module_name: str, attrs: tuple[str, ...]) -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc


        class _BlockedInferenceDependencyFinder(importlib.abc.MetaPathFinder):
            _BLOCKED = {{"torch", "transformers", "safetensors"}}

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in self._BLOCKED:
                    raise ModuleNotFoundError(f"Blocked import for test: {{fullname}}")
                return None


        import sys

        sys.meta_path.insert(0, _BlockedInferenceDependencyFinder())
        module_name = {module_name!r}
        module = importlib.import_module(module_name)
        for attr in {attrs!r}:
            if not hasattr(module, attr):
                raise SystemExit("Missing " + module_name + "." + attr)
        """
    )
    result = _run_subprocess([sys.executable, "-c", script])
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


class PackagingCLITests(unittest.TestCase):
    def test_pyproject_declares_expected_console_scripts(self) -> None:
        scripts = _parse_project_scripts(PROJECT_ROOT / "pyproject.toml")
        self.assertEqual(scripts.get("ai-detector"), "run_ensemble:main")
        self.assertEqual(scripts.get("ai-detector-deploy"), "deploy_meld:main")

    def test_modules_importable_without_inference_dependencies(self) -> None:
        _run_import_smoke(
            "run_ensemble",
            ("parse_args", "main", "load_meld", "load_tmr", "load_raid"),
        )
        _run_import_smoke("deploy_meld", ("parse_args", "main"))

    def test_parse_project_scripts_fallback_parses_quoted_keys_and_inline_comments(self) -> None:
        content = textwrap.dedent(
            """
            [project.scripts] # entry points
            "ai-detector" = "run_ensemble:main" # primary
            'ai-detector-deploy' = 'deploy_meld:main'  # deploy entry
            ai-with-hash = "path#not-a-comment" # inline with hash in value
            """
        )

        with tempfile.NamedTemporaryFile("w", suffix=".toml", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            parsed = _parse_project_scripts_fallback(Path(handle.name).read_text(encoding="utf-8"))

        self.assertEqual(parsed["ai-detector"], "run_ensemble:main")
        self.assertEqual(parsed["ai-detector-deploy"], "deploy_meld:main")
        self.assertEqual(parsed["ai-with-hash"], "path#not-a-comment")

    def test_run_ensemble_help_smoke(self) -> None:
        result = _run_subprocess([sys.executable, str(PROJECT_ROOT / "run_ensemble.py"), "--help"])
        self.assertEqual(result.returncode, 0)
        output = (result.stdout + result.stderr).lower()
        self.assertIn("usage:", output)
        self.assertIn("--text", output)
        self.assertIn("--text-file", output)
        self.assertIn("--quiet", output)
        self.assertIn("--json", output)

    def test_deploy_meld_help_smoke(self) -> None:
        result = _run_subprocess([sys.executable, str(PROJECT_ROOT / "deploy_meld.py"), "--help"])
        self.assertEqual(result.returncode, 0)
        output = (result.stdout + result.stderr).lower()
        self.assertIn("usage:", output)
        self.assertIn("--target-dir", output)
        self.assertIn("--model-id", output)
        self.assertIn("--revision", output)

    def test_main_json_contract_subprocess(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import run_ensemble
            import sys

            fake_result = {
                "text_preview": "hello",
                "weights": {"meld": 0.34, "tmr": 0.33, "raid": 0.33},
                "experts": {
                    "meld": {
                        "ai_score": 0.12,
                        "human_score": 0.88,
                        "ai_probability": 0.12,
                        "human_probability": 0.88,
                        "chunks": 1,
                        "loaded": True,
                    },
                    "tmr": {
                        "ai_score": 0.22,
                        "human_score": 0.78,
                        "ai_probability": 0.22,
                        "human_probability": 0.78,
                        "chunks": 1,
                        "loaded": True,
                    },
                    "raid": {
                        "ai_score": 0.32,
                        "human_score": 0.68,
                        "ai_probability": 0.32,
                        "human_probability": 0.68,
                        "chunks": 1,
                        "loaded": True,
                    },
                },
                "ensemble": {
                    "ai_score": 0.22,
                    "human_score": 0.78,
                    "ai_probability": 0.22,
                    "human_probability": 0.78,
                    "threshold": 0.5,
                    "label": "human",
                },
                "calibration": {
                    "status": "uncalibrated",
                    "calibrated": False,
                    "message": "test",
                },
                "device": "cpu",
            }

            def _fake_run_ensemble(text, args):
                return fake_result

            run_ensemble.run_ensemble = _fake_run_ensemble
            sys.argv = ["run_ensemble.py", "--json", "--text", "hello"]
            run_ensemble.main()
            """
        )
        result = _run_subprocess([sys.executable, "-c", script])

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        for key in ("experts", "ensemble", "calibration", "weights", "device"):
            self.assertIn(key, payload)
        self.assertIn("label", payload["ensemble"])
        for expert in ("meld", "tmr", "raid"):
            self.assertIn(expert, payload["experts"])
            for key in (
                "ai_score",
                "human_score",
                "ai_probability",
                "human_probability",
                "chunks",
                "loaded",
            ):
                self.assertIn(key, payload["experts"][expert])
        for key in (
            "ai_probability",
            "human_probability",
            "threshold",
            "label",
        ):
            self.assertIn(key, payload["ensemble"])
        for key in ("status", "calibrated", "message"):
            self.assertIn(key, payload["calibration"])
