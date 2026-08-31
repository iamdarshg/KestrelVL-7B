"""Resumable ranged downloader for the real Nemotron checkpoint.

The HF CDN occasionally closes very large single requests on Windows. This
script downloads each public shard in bounded ranges and leaves ``.part``
files for exact resume. The destination is ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def resolve(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"


def download(repo: str, filename: str, revision: str, destination: Path, chunk_size: int) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = resolve(repo, filename, revision)
    for attempt in range(8):
        try:
            head = requests.head(url, allow_redirects=True, timeout=60)
            head.raise_for_status()
            content_length = head.headers.get("content-length")
            if content_length is None:
                response = requests.get(url, allow_redirects=True, timeout=(60, 180))
                response.raise_for_status()
                destination.write_bytes(response.content)
                return {
                    "file": filename,
                    "bytes": len(response.content),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                }
            total = int(content_length)
            part = destination.with_suffix(destination.suffix + ".part")
            current = part.stat().st_size if part.exists() else 0
            if current > total:
                part.unlink()
                current = 0
            with part.open("ab") as handle:
                while current < total:
                    end = min(total - 1, current + chunk_size - 1)
                    response = requests.get(
                        url,
                        headers={"Range": f"bytes={current}-{end}"},
                        allow_redirects=True,
                        stream=True,
                        timeout=(60, 180),
                    )
                    response.raise_for_status()
                    content_range = response.headers.get("content-range", "")
                    if not content_range.startswith(f"bytes {current}-{end}/"):
                        raise RuntimeError(f"unexpected content-range for {filename}: {content_range}")
                    expected = end - current + 1
                    written = 0
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                            written += len(block)
                    if written != expected:
                        raise IOError(f"short ranged response for {filename}: {written} != {expected}")
                    current = end + 1
                    handle.flush()
                    print(f"{filename}: {current}/{total} ({current / total:.1%})", flush=True)
            part.replace(destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            return {"file": filename, "bytes": total, "sha256": digest}
        except Exception as error:
            print(f"retry {attempt + 1}/8 for {filename}: {error}", flush=True)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to download {filename} after retries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="nvidia/OpenReasoning-Nemotron-7B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="data/raw/nemotron")
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "files": [download(args.repo, filename, args.revision, output / filename, args.chunk_mib * 2**20) for filename in FILES],
    }
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
