#!/usr/bin/env python3
"""Download Hugging Face model files into the working directory."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MODEL_ID = "anon-review-meld-2026/meld"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}"
RESOLVE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/"
CHUNK_SIZE = 1024 * 256


def _human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def file_needs_update(dst: Path, expected_size: int | None) -> bool:
    if not dst.exists():
        return True
    if expected_size is None:
        return False
    return dst.stat().st_size != expected_size


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "ai-detector-deployer/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        expected_size = response.headers.get("Content-Length")
        expected_size_int = int(expected_size) if expected_size is not None else None

        if expected_size_int is not None and not file_needs_update(dst, expected_size_int):
            print(f"✓ {dst} already up to date ({_human_bytes(expected_size_int)})")
            return

        bytes_total = int(expected_size) if expected_size else 0
        with dst.open("wb") as out_file:
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
            if bytes_total and downloaded != bytes_total:
                raise RuntimeError(
                    f"Incomplete download for {url}. Expected {bytes_total} bytes, got {downloaded}."
                )

    if bytes_total:
        print(f"⬇ {dst} ({_human_bytes(bytes_total)})")
    else:
        print(f"⬇ {dst}")


def download_model(target_dir: Path) -> None:
    data = fetch_json(API_URL)
    siblings = data.get("siblings", [])
    if not siblings:
        raise RuntimeError("No files found in model metadata.")

    for sibling in siblings:
        filename = sibling["rfilename"]
        encoded = urllib.parse.quote(filename)
        url = f"{RESOLVE_URL}{encoded}"
        dst = target_dir / filename

        try:
            download_file(url, dst)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RuntimeError(
                    "Rate-limited while downloading files. Retry after waiting a bit."
                ) from exc
            raise RuntimeError(
                f"Failed to download {filename}: HTTP {exc.code} {exc.reason}"
            ) from exc


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target_dir
    global MODEL_ID, API_URL, RESOLVE_URL
    MODEL_ID = args.model_id
    API_URL = f"https://huggingface.co/api/models/{MODEL_ID}"
    RESOLVE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/"

    print(f"Deploying {MODEL_ID} to {target.resolve()}")
    download_model(target)
    print("Deployment complete.")


if __name__ == "__main__":
    main()
