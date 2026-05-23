"""Unit tests for deploy_meld download validation flow."""

from __future__ import annotations

import io
import tempfile
import textwrap
from pathlib import Path
import os
import subprocess
import sys
from unittest import TestCase
from unittest.mock import patch

import deploy_meld


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _MockHTTPResponse:
    def __init__(self, content_length: str | None, body: bytes = b"") -> None:
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self._body = io.BytesIO(body)

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


class DeployMeldTests(TestCase):
    def test_main_prints_completion_message_and_calls_download_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            model_id = "owner/custom-meld"
            revision = "v2"
            argv = [
                "deploy_meld.py",
                "--model-id",
                model_id,
                "--revision",
                revision,
                "--target-dir",
                str(target_dir),
            ]

            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.object(sys, "argv", argv):

                def fake_download_model(
                    observed_model_id: str, observed_revision: str, observed_target_dir: Path
                ) -> list[dict]:
                    self.assertEqual(observed_model_id, model_id)
                    self.assertEqual(observed_revision, revision)
                    self.assertEqual(observed_target_dir, target_dir)
                    self.assertNotIn("Deployment complete.", stdout.getvalue())
                    return []

                with patch.object(
                    deploy_meld, "download_model", side_effect=fake_download_model
                ) as mocked_download, patch("sys.stdout", new=stdout), patch("sys.stderr", new=stderr):
                    deploy_meld.main()

            mocked_download.assert_called_once_with(model_id, revision, target_dir)
            self.assertIn(
                f"Deploying {model_id} @ {revision} to {target_dir.resolve()}",
                stdout.getvalue(),
            )
            self.assertIn("Deployment complete.", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_file_needs_update_true_when_expected_size_missing_and_head_size_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "file.bin"
            target.write_bytes(b"12345")

            with patch("deploy_meld.urllib.request.urlopen") as mocked_urlopen:
                mocked_urlopen.return_value = _MockHTTPResponse("10")
                needs_update, _ = deploy_meld._file_needs_update(target, None, "https://example.com/file.bin")

            self.assertTrue(needs_update)

    def test_download_model_updates_when_head_content_length_is_higher_than_existing_file(self) -> None:
        metadata = {"siblings": [{"rfilename": "file.bin"}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            existing = target_dir / "file.bin"
            existing.write_bytes(b"12345")

            call_order: list[str] = []

            def fake_urlopen(request, timeout: int = 60):
                method = request.get_method()
                call_order.append(method)

                if method == "HEAD":
                    return _MockHTTPResponse("10")
                if method == "GET":
                    return _MockHTTPResponse("10", body=b"x" * 10)
                raise AssertionError(f"Unexpected method {method}")

            with patch.object(deploy_meld, "fetch_json", return_value=metadata):
                with patch("deploy_meld.urllib.request.urlopen", side_effect=fake_urlopen):
                    downloaded = deploy_meld.download_model("owner/repo", "main", target_dir)

            self.assertEqual(call_order[0], "HEAD")
            self.assertEqual(call_order[1], "GET")
            self.assertEqual(existing.read_bytes(), b"x" * 10)
            self.assertEqual(downloaded[0]["size"], 10)

    def test_download_model_preserves_nested_filename_slashes_in_urls(self) -> None:
        metadata = {"siblings": [{"rfilename": "nested/file.bin"}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            existing = target_dir / "nested" / "file.bin"
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"1")

            call_urls: list[tuple[str, str]] = []

            def fake_urlopen(request, timeout: int = 60):
                method = request.get_method()
                call_urls.append((method, request.full_url))

                if method == "HEAD":
                    return _MockHTTPResponse("5")
                if method == "GET":
                    return _MockHTTPResponse("5", body=b"x" * 5)
                raise AssertionError(f"Unexpected method {method}")

            with patch.object(deploy_meld, "fetch_json", return_value=metadata):
                with patch("deploy_meld.urllib.request.urlopen", side_effect=fake_urlopen):
                    downloaded = deploy_meld.download_model("owner/repo", "main", target_dir)

            head_urls = [url for method, url in call_urls if method == "HEAD"]
            get_urls = [url for method, url in call_urls if method == "GET"]
            self.assertEqual(len(head_urls), 1)
            self.assertEqual(len(get_urls), 1)
            self.assertIn("/nested/file.bin", head_urls[0])
            self.assertIn("/nested/file.bin", get_urls[0])
            self.assertNotIn("nested%2Ffile.bin", head_urls[0])
            self.assertNotIn("nested%2Ffile.bin", get_urls[0])
            self.assertTrue((target_dir / "nested" / "file.bin").exists())
            self.assertEqual(existing.read_bytes(), b"x" * 5)
            self.assertEqual(downloaded[0]["path"], "nested/file.bin")

    def test_main_turns_runtime_error_into_operator_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = deploy_meld.argparse.Namespace(
                target_dir=Path(tmpdir), model_id="owner/repo", revision="main"
            )

            with patch.object(deploy_meld, "parse_args", return_value=args), patch.object(
                deploy_meld, "download_model", side_effect=RuntimeError("boom")
            ), patch("sys.stdout", new=io.StringIO()) as stdout, patch(
                "sys.stderr", new=io.StringIO()
            ) as stderr:
                with self.assertRaises(SystemExit) as exc:
                    deploy_meld.main()

            self.assertEqual(exc.exception.code, 1)
            self.assertIn("Error: boom", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn("Deploying owner/repo @ main", stdout.getvalue())

    def test_main_turns_value_error_into_operator_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = deploy_meld.argparse.Namespace(
                target_dir=Path(tmpdir), model_id="owner/repo", revision="main"
            )

            with patch.object(deploy_meld, "parse_args", return_value=args), patch.object(
                deploy_meld, "download_model", side_effect=ValueError("bad revision")
            ), patch("sys.stdout", new=io.StringIO()) as stdout, patch(
                "sys.stderr", new=io.StringIO()
            ) as stderr:
                with self.assertRaises(SystemExit) as exc:
                    deploy_meld.main()

            self.assertEqual(exc.exception.code, 1)
            self.assertIn("Error: bad revision", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn("Deploying owner/repo @ main", stdout.getvalue())

    def test_main_turns_os_error_into_operator_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = deploy_meld.argparse.Namespace(
                target_dir=Path(tmpdir), model_id="owner/repo", revision="main"
            )

            with patch.object(deploy_meld, "parse_args", return_value=args), patch.object(
                deploy_meld, "download_model", side_effect=OSError("disk unavailable")
            ), patch("sys.stdout", new=io.StringIO()) as stdout, patch(
                "sys.stderr", new=io.StringIO()
            ) as stderr:
                with self.assertRaises(SystemExit) as exc:
                    deploy_meld.main()

            self.assertEqual(exc.exception.code, 1)
            self.assertIn("Error: disk unavailable", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn("Deploying owner/repo @ main", stdout.getvalue())

    def test_main_expected_failure_exits_with_code_1_without_traceback(self) -> None:
        script = textwrap.dedent(
            """
            import sys
            import tempfile
            import unittest.mock

            import deploy_meld


            with tempfile.TemporaryDirectory() as tmpdir:
                argv = [
                    "deploy_meld.py",
                    "--model-id",
                    "owner/repo",
                    "--revision",
                    "main",
                    "--target-dir",
                    tmpdir,
                ]
                with unittest.mock.patch.object(
                    deploy_meld, "download_model", side_effect=RuntimeError("subprocess boom")
                ):
                    sys.argv = argv
                    deploy_meld.main()
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            timeout=10,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("Deploying owner/repo @ main", completed.stdout)
        self.assertIn("Error: subprocess boom", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
