"""Resumable ranged downloader for the real Nemotron checkpoint.

The HF CDN occasionally closes very large single requests on Windows. This
script downloads each public shard in bounded ranges and leaves ``.part``
files for exact resume. The destination is ignored by git.
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                    "sha256": sha256_file(destination),
                }
            total = int(content_length)
            part = destination.with_suffix(destination.suffix + ".part")
            prefix = part.stat().st_size if part.exists() else 0
            if prefix > total:
                part.unlink()
                prefix = 0
            ranges = destination.parent / (destination.name + ".ranges")
            ranges.mkdir(exist_ok=True)
            if prefix == total:
                # This also finalizes a file completed by aria2c, whose
                # resumable output uses the same `.part` convention.
                part.replace(destination)
                for child in ranges.iterdir():
                    if child.is_file():
                        child.unlink()
                ranges.rmdir()
                return {"file": filename, "bytes": total, "sha256": sha256_file(destination)}
            starts = list(range(prefix, total, chunk_size))

            def download_range(start: int) -> tuple[int, int]:
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
                        expected_range = f"bytes {start + current}-{end}/"
                        if not content_range.startswith(expected_range):
                            raise RuntimeError(f"unexpected content-range for {filename}: {content_range}")
                        with target.open("ab") as handle:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    handle.write(block)
                        current = target.stat().st_size
                        if current != expected:
                            raise IOError(f"short ranged response for {filename}: {current} != {expected}")
                    except Exception as error:
                        print(f"retry {retry + 1}/8 for {filename} range {start}: {error}", flush=True)
                        time.sleep(min(30, 2**retry))
                raise RuntimeError(f"failed range {start}-{end} for {filename}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(download_range, start) for start in starts]
                for future in concurrent.futures.as_completed(futures):
                    start, end = future.result()
                    print(f"{filename}: range {start}-{end} complete", flush=True)
            with part.open("ab") as handle:
                current = prefix
                for start in starts:
                    target = ranges / f"{start:012d}.part"
                    if start < current:
                        continue
                    if target.stat().st_size != min(total, start + chunk_size) - start:
                        raise IOError(f"incomplete range file for {filename}: {target}")
                    with target.open("rb") as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                    current += target.stat().st_size
                    target.unlink()
                    print(f"{filename}: assembled {current}/{total} ({current / total:.1%})", flush=True)
            part.replace(destination)
            ranges.rmdir()
            return {"file": filename, "bytes": total, "sha256": sha256_file(destination)}
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
