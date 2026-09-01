"""Create and validate a pickle-free Kestrel Q4 release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release.serialization import load_q4_bundle, save_q4_bundle  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def validate(root: Path) -> None:
    required = ["model.safetensors", "config.json", "quantization_config.json", "checksums.json", "kestrel_runtime.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"release validation failed; missing: {missing}")
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    observed = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
    if checksums.get("model.safetensors") != observed:
        raise SystemExit("release validation failed; model.safetensors checksum mismatch")
    state = load_q4_bundle(root)
    if not state:
        raise SystemExit("release validation failed; no tensors")
    manifest = json.loads((root / "quantization_config.json").read_text(encoding="utf-8"))
    if manifest.get("pickle_free_loader") is not True:
        raise SystemExit("release validation failed; pickle-free marker absent")
    print(json.dumps({"valid": True, "tensors": len(state), "format": manifest["format"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-safetensors", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate is not None:
        validate(args.validate)
        return
    if args.state_safetensors is None or args.output is None:
        parser.error("--state-safetensors and --output are required for export")
    state = load_file(str(args.state_safetensors), device="cpu")
    config = {}
    if args.config_json is not None:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
    save_q4_bundle(state, args.output, config, group_size=args.group_size, force=args.force)
    # A fresh interpreter is part of the export gate: this catches accidental
    # reliance on an in-process object or an unsafe pickle loader.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--validate", str(args.output)], check=True)
    print(f"release={args.output.resolve()}")


if __name__ == "__main__":
    main()
