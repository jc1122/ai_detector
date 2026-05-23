"""Tests for packaging and CLI entrypoint contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
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

    scripts: dict[str, str] = {}
    in_scripts_block = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            in_scripts_block = section == "project.scripts"
            continue
        if not in_scripts_block or "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
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
        _run_import_smoke("run_ensemble", ("parse_args", "main", "load_meld", "load_tmr"))
        _run_import_smoke("deploy_meld", ("parse_args", "main"))

    def test_run_ensemble_help_smoke(self) -> None:
        result = _run_subprocess([sys.executable, str(PROJECT_ROOT / "run_ensemble.py"), "--help"])
        self.assertEqual(result.returncode, 0)
        output = (result.stdout + result.stderr).lower()
        self.assertIn("usage:", output)
        self.assertIn("--text", output)
        self.assertIn("--text-file", output)
        self.assertIn("--json", output)

    def test_deploy_meld_help_smoke(self) -> None:
        result = _run_subprocess([sys.executable, str(PROJECT_ROOT / "deploy_meld.py"), "--help"])
        self.assertEqual(result.returncode, 0)
        output = (result.stdout + result.stderr).lower()
        self.assertIn("usage:", output)
        self.assertIn("--target-dir", output)
        self.assertIn("--model-id", output)
        self.assertIn("--revision", output)
