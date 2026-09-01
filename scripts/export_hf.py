"""Export a safetensors-backed HF-compatible Kestrel state/config pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import KestrelConfig  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-safetensors", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="legacy pickle checkpoint; rejected for safe export")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.checkpoint is not None:
        raise SystemExit("HF export blocked for pickle checkpoint; convert from --state-safetensors")
    if args.state_safetensors is None:
        parser.error("--state-safetensors is required")
    args.output.mkdir(parents=True, exist_ok=True)
    state = load_file(str(args.state_safetensors), device="cpu")
    save_file(state, str(args.output / "model.safetensors"), metadata={"format": "kestrel-hf-v1"})
    if args.config_json is not None:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
    else:
        config = KestrelConfig.tiny(use_vision=False).to_dict()
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"hf_export={args.output.resolve()}")


if __name__ == "__main__":
    main()
