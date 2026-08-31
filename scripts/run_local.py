"""Dependency-light local inference smoke path for text and screenshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from model import KestrelConfig, KestrelForCausalLM


def encode(text: str, vocab_size: int) -> torch.Tensor:
    return torch.tensor([[ord(char) % vocab_size for char in text]], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--model-path", help="reserved for an exported checkpoint; default uses tiny smoke weights")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.model_path:
        raise NotImplementedError("load an exported HF checkpoint through the stage exporter; smoke weights are used when omitted")
    use_vision = bool(args.image)
    config = KestrelConfig.tiny(use_vision=use_vision)
    model = KestrelForCausalLM(config).eval()
    pixels = None
    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        pixels = torch.from_numpy(__import__("numpy").asarray(image)).permute(2, 0, 1).float() / 255
    ids = encode(args.prompt, config.vocab_size)
    output = model.generate(ids, args.max_new_tokens, pixel_values=pixels)
    print("token_ids:", output[0].tolist())
    print("note: this bootstrap smoke path uses untrained tiny weights; it is not a quality result")


if __name__ == "__main__":
    main()

