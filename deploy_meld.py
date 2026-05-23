#!/usr/bin/env python3
"""Download Hugging Face model files into the working directory."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
import urllib
import urllib.error
import urllib.parse
import urllib.request
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "anon-review-meld-2026/meld"
REVISION = "main"
MANIFEST_FILENAME = "ai_detector_model_manifest.json"
CHUNK_SIZE = 1024 * 256


def _human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024


def _safe_error(context: str, exc: Exception) -> RuntimeError:
    if isinstance(exc, urllib.error.HTTPError):
        return RuntimeError(
            f"{context}: HTTP {exc.code} {exc.reason} for {exc.url}"
        )
    if isinstance(exc, urllib.error.URLError):
        return RuntimeError(f"{context}: {exc.reason}")
    return RuntimeError(f"{context}: {exc}")


def _build_model_url(model_id: str, revision: str) -> tuple[str, str]:
    quoted_model_id = "/".join(urllib.parse.quote(part, safe="") for part in model_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    return (
        f"https://huggingface.co/api/models/{quoted_model_id}?revision={quoted_revision}",
        f"https://huggingface.co/{quoted_model_id}/resolve/{quoted_revision}/",
    )


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            try:
                return json.load(resp)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError(f"Invalid JSON response from {url}") from exc
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise _safe_error(f"Failed to fetch metadata from {url}", exc) from exc


def _parse_content_length(content_length: str | None) -> int | None:
    if not content_length:
        return None
    try:
        return int(content_length)
    except (TypeError, ValueError):
        return None


def _fetch_remote_size(url: str) -> int | None:
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ai-detector-deployer/1.0"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return _parse_content_length(resp.headers.get("Content-Length"))
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code == 405:
                # Some endpoints reject HEAD; fallback to a lightweight GET header-only check.
                continue
            raise _safe_error(f"Failed to fetch remote size from {url}", exc)

        except urllib.error.URLError as exc:
            raise _safe_error(f"Failed to fetch remote size from {url}", exc)

    return None


def _file_needs_update(dst: Path, expected_size: int | None, url: str | None) -> tuple[bool, int | None]:
    if not dst.exists():
        return True, expected_size
    if expected_size is None:
        if url is None:
            return True, None

        remote_size = _fetch_remote_size(url)
        if remote_size is None:
            return False, None
        return dst.stat().st_size != remote_size, remote_size
    return dst.stat().st_size != expected_size, expected_size


def _safe_destination(target_dir: Path, rfilename: str) -> Path:
    if not rfilename or rfilename in {".", ".."}:
        raise RuntimeError(f"Invalid model file path: {rfilename!r}")
    if rfilename.startswith(("/", "\\")):
        raise RuntimeError(f"Unsafe absolute path in model file: {rfilename!r}")
    if "\\" in rfilename:
        raise RuntimeError(f"Unsafe path separator in model file: {rfilename!r}")

    normalized = Path(*rfilename.split("/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise RuntimeError(f"Path traversal attempt detected: {rfilename!r}")

    target_root = target_dir.resolve()
    target_path = (target_root / normalized).resolve()
    if os.path.commonpath([str(target_root), str(target_path)]) != str(target_root):
        raise RuntimeError(f"Path would escape target directory: {rfilename!r}")

    return target_root / normalized


def download_file(url: str, dst: Path, expected_size: int | None = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    needs_update, remote_size = _file_needs_update(dst, expected_size, url)
    if not needs_update:
        size = dst.stat().st_size
        print(f"✓ {dst} already up to date ({_human_bytes(size)})")
        return size

    temp_fd, temp_path = tempfile.mkstemp(dir=dst.parent, prefix=".ai-detector-", suffix=".tmp")
    os.close(temp_fd)
    temp_dst = Path(temp_path)

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai-detector-deployer/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                size_header = response.headers.get("Content-Length")
                expected_from_header = _parse_content_length(size_header)
                effective_size = expected_size if expected_size is not None else remote_size
                if expected_from_header is not None:
                    effective_size = expected_from_header

                downloaded = 0
                with temp_dst.open("wb") as out_file:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)

            if effective_size is not None and downloaded != effective_size:
                raise RuntimeError(
                    f"Incomplete download for {url}. Expected {effective_size} bytes, got {downloaded}."
                )
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise _safe_error(f"Failed to download {url}", exc)

        temp_size = downloaded
        temp_dst.replace(dst)

        if temp_size:
            print(f"⬇ {dst} ({_human_bytes(temp_size)})")
        else:
            print(f"⬇ {dst}")

        return temp_size
    finally:
        if temp_dst.exists():
            temp_dst.unlink()


def download_model(model_id: str, revision: str, target_dir: Path) -> list[dict[str, Any]]:
    api_url, resolve_base = _build_model_url(model_id, revision)
    data = fetch_json(api_url)
    siblings = data.get("siblings", [])
    if not siblings:
        raise RuntimeError("No files found in model metadata.")

    downloaded_files = []
    target_root = target_dir.resolve()
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue

        filename = sibling.get("rfilename")
        if not isinstance(filename, str):
            continue

        dst = _safe_destination(target_dir, filename)
        encoded = "/".join(
            urllib.parse.quote(part, safe="") for part in filename.split("/")
        )
        url = f"{resolve_base}{encoded}"
        expected_size = sibling.get("size")
        expected_size_int = int(expected_size) if isinstance(expected_size, int) else None

        size = download_file(url, dst, expected_size_int)

        downloaded_files.append(
            {
                "path": str(dst.relative_to(target_root)),
                "rfilename": filename,
                "size": size,
                "metadata": sibling,
            }
        )

    manifest_metadata = {
        key: value
        for key, value in data.items()
        if key != "siblings" and value is not None
    }
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": manifest_metadata,
        "files": downloaded_files,
        "pliki": [entry["path"] for entry in downloaded_files],
        "rozmiary": {entry["path"]: entry.get("size") for entry in downloaded_files},
    }

    manifest_path = target_dir / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🧾 Wrote manifest to {manifest_path}")

    return downloaded_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model files into the working directory."
    )
    parser.add_argument(
        "--target-dir",
        default=Path("models/meld"),
        type=Path,
        help="Directory to store downloaded model files.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model id (owner/repo).",
    )
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="Model revision used in API and file download URL (default: main).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        target = args.target_dir
        model_id = args.model_id
        revision = args.revision
        print(f"Deploying {model_id} @ {revision} to {target.resolve()}")
        download_model(model_id, revision, target)
        print("Deployment complete.")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
