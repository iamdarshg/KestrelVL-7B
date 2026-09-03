"""Validate and fingerprint the real corpus configuration without consuming data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.real_corpus import RealCorpusSpec, write_canonical_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/real_corpus.yaml")
    parser.add_argument("--output", default="data/real_corpus_manifest.json")
    args = parser.parse_args()
    spec = RealCorpusSpec.from_yaml(ROOT / args.config)
    fingerprint = write_canonical_manifest(spec, ROOT / args.output)
    print(json.dumps({"status": "pass", "fingerprint": fingerprint, "config": args.config}, indent=2))


if __name__ == "__main__":
    main()
