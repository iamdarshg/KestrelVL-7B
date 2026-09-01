"""Download the immutable InternViT developer-vision reference.

The checkpoint is kept outside Git.  ``snapshot_download`` is resumable at the
file level and preserves the remote-code files required by InternViT.  The
resulting directory can be passed directly to ``graft_nemotron.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="OpenGVLab/InternViT-300M-448px-V2_5")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="data/raw/internvit")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(output),
        allow_patterns=(
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "*.safetensors",
            "*.bin",
            "*.py",
        ),
    )
    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "path": str(output),
        "note": "Weights are external artifacts; verify the HF revision before training.",
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
