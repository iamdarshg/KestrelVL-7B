"""Resumably download the Google TIPSv2 L/14 vision checkpoint.

The large safetensors file is downloaded in independently resumable byte
ranges.  This avoids relying on one long-lived CDN request and leaves useful
progress after a laptop sleep, network reset, or preemptible worker shutdown.
The downloaded directory is ignored by Git; ``download_manifest.json`` records
the exact revision, sizes, and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests


FILES = (
    "config.json",
    "configuration_tips.py",
    "image_encoder.py",
    "modeling_tips.py",
    "processor_config.json",
    "text_encoder.py",
    "tokenizer.model",
    "tokenizer_config.json",
    "model.safetensors",
)


def resolve(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    length = response.headers.get("content-length")
    if length is None:
        raise RuntimeError(f"content length missing for {url}")
    return int(length)


def download_file(repo: str, filename: str, revision: str, destination: Path, chunk_size: int) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = resolve(repo, filename, revision)
    total = _head(url)
    if destination.exists() and destination.stat().st_size == total:
        return {"file": filename, "bytes": total, "sha256": sha256_file(destination), "resumed": False}

    part = destination.with_suffix(destination.suffix + ".part")
    ranges = destination.parent / (destination.name + ".ranges")
    ranges.mkdir(exist_ok=True)
    prefix = part.stat().st_size if part.exists() else 0
    if prefix > total:
        part.unlink()
        prefix = 0
    starts = list(range(prefix, total, chunk_size))

    def fetch(start: int) -> tuple[int, int]:
        end = min(total - 1, start + chunk_size - 1)
        target = ranges / f"{start:012d}.part"
        current = target.stat().st_size if target.exists() else 0
        expected = end - start + 1
        if current > expected:
            target.unlink()
            current = 0
        for retry in range(8):
            if current == expected:
                return start, end
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={start + current}-{end}"},
                    allow_redirects=True,
                    stream=True,
                    timeout=(60, 180),
                )
                response.raise_for_status()
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {start + current}-{end}/"):
                    raise RuntimeError(f"unexpected content-range: {content_range}")
                with target.open("ab") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                current = target.stat().st_size
                if current != expected:
                    raise IOError(f"short response: {current} != {expected}")
            except Exception as error:
                print(f"retry {retry + 1}/8 {filename} range {start}: {error}", flush=True)
                time.sleep(min(30, 2**retry))
        raise RuntimeError(f"failed {filename} range {start}-{end}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for future in concurrent.futures.as_completed(futures):
            start, end = future.result()
            print(f"{filename}: range {start}-{end} complete", flush=True)

    current = prefix
    with part.open("ab") as handle:
        for start in starts:
            target = ranges / f"{start:012d}.part"
            expected = min(total, start + chunk_size) - start
            if target.stat().st_size != expected:
                raise IOError(f"incomplete range file: {target}")
            with target.open("rb") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            current += expected
            target.unlink()
            print(f"{filename}: assembled {current}/{total} ({current / total:.1%})", flush=True)
    part.replace(destination)
    if not any(ranges.iterdir()):
        ranges.rmdir()
    return {"file": filename, "bytes": total, "sha256": sha256_file(destination), "resumed": prefix > 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="google/tipsv2-l14")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="data/raw/tipsv2-l14")
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "files": [
            download_file(args.repo, filename, args.revision, output / filename, args.chunk_mib * 2**20)
            for filename in FILES
        ],
    }
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
