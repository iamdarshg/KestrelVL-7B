"""Report analytical cache memory at the requested context lengths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.long_context import estimate_cache_memory
from model.configuration import KestrelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[4096, 32768, 131072, 1048576, 1500000],
    )
    parser.add_argument("--index-topk", type=int, default=256)
    parser.add_argument("--index-dtype", choices=["int8", "int16", "int32", "int64", "bfloat16"], default="int8")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = KestrelConfig(
        index_topk=args.index_topk,
        index_dtype=args.index_dtype,
        use_vision=False,
    )
    report = {
        "architecture": config.to_dict(),
        "estimates": [estimate_cache_memory(config, context) for context in args.contexts],
        "evidence_label": "analytical_cache_state_estimate_not_measured_inference",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
