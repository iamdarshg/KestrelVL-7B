"""Load and image-smoke a local InternViT or TIPSv2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.vision.internvit import InternViTEncoder, dynamic_tiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image")
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--trainable-smoke", action="store_true")
    args = parser.parse_args()

    path = Path(args.model_path)
    if not (path / "config.json").exists():
        raise SystemExit(f"missing config.json: {path}")
    if not ((path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()):
        raise SystemExit(f"no complete model weight file found: {path}")
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    encoder = InternViTEncoder(model_path=path, freeze=not args.trainable_smoke, torch_dtype=dtype)
    if args.image:
        from PIL import Image
        import numpy as np

        image = Image.open(args.image).convert("RGB")
        pixels = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    else:
        pixels = torch.rand(3, 448, 448)
    pixels = dynamic_tiles(pixels, max_tiles=args.max_tiles)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(args.trainable_smoke):
        tokens = encoder.encode_with_policy(pixels.to(device), device, offload_to_cpu=False)
    result = {
        "path": str(path.resolve()),
        "backend": encoder.backend,
        "hidden_size": encoder.hidden_size,
        "tiles": int(pixels.shape[0]),
        "tokens": int(tokens.shape[1]),
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "finite": bool(torch.isfinite(tokens).all()),
        "telemetry": encoder.last_telemetry,
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    if not result["finite"]:
        raise FloatingPointError("vision checkpoint emitted non-finite tokens")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
