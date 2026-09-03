"""Validate the governed corpus contract without downloading full datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.real_corpus import RealCorpusSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/real_corpus.yaml")
    parser.add_argument("--require-local-content", action="store_true")
    args = parser.parse_args()
    spec = RealCorpusSpec.from_yaml(ROOT / args.config)
    sources = []
    missing = []
    for source in spec.source_specs:
        record = {"name": source.name, "kind": source.kind, "uri": source.uri, "revision": source.revision}
        if source.kind in {"jsonl", "jsonl.gz"}:
            path = Path(source.uri)
            if not path.is_absolute():
                path = ROOT / path
            record["path"] = str(path)
            record["exists"] = path.is_file()
            if not path.is_file():
                missing.append(f"{source.name}: {path}")
        else:
            record["streaming"] = True
            record["content_resolver"] = source.content_resolver
        sources.append(record)
    report = {
        "status": "fail" if missing and args.require_local_content else "pass",
        "config": args.config,
        "fingerprint": spec.fingerprint(),
        "sources": sources,
        "missing_local_content": missing,
        "network_probe": False,
        "note": "This command validates identity and local availability only; it never downloads a corpus.",
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
