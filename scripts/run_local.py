"""Dependency-light local inference smoke path for text and screenshots."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from model import KestrelConfig, KestrelForCausalLM
from release.serialization import load_q4_bundle
from safetensors.torch import load_file


def encode(text: str, vocab_size: int) -> torch.Tensor:
    return torch.tensor([[ord(char) % vocab_size for char in text]], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--model-path", help="reserved for an exported checkpoint; default uses tiny smoke weights")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    use_vision = bool(args.image)
    state = None
    model_note = "bootstrap smoke path uses untrained tiny weights"
    if args.model_path:
        root = Path(args.model_path)
        config_path = root / "config.json"
        if not config_path.exists():
            raise SystemExit(f"model path has no config.json: {root}")
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        fields = {field.name for field in dataclasses.fields(KestrelConfig)}
        config = KestrelConfig(**{key: value for key, value in raw_config.items() if key in fields})
        if (root / "quantization_config.json").exists():
            state = load_q4_bundle(root, device="cpu")
            model_note = "loaded pickle-free Kestrel Q4 bundle; runtime dequantizes bootstrap tensors"
        elif (root / "model.safetensors").exists():
            state = load_file(str(root / "model.safetensors"), device="cpu")
            model_note = "loaded safetensors Kestrel state"
        else:
            raise SystemExit(f"model path has no model.safetensors or quantization_config.json: {root}")
    else:
        config = KestrelConfig.tiny(use_vision=use_vision)
    device = torch.device(args.device)
    model = KestrelForCausalLM(config)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"state/config mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model = model.to(device).eval()
    pixels = None
    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        pixels = torch.from_numpy(__import__("numpy").asarray(image)).permute(2, 0, 1).float() / 255
    ids = encode(args.prompt, config.vocab_size).to(device)
    output = model.generate(ids, args.max_new_tokens, pixel_values=pixels)
    print("token_ids:", output[0].tolist())
    print(f"note: {model_note}; this is not a quality result")


if __name__ == "__main__":
    main()
