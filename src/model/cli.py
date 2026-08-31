"""One-command local smoke inference for text, images, and tool schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .configuration import KestrelConfig
from .multimodal_model import KestrelForCausalLM


def _encode(text: str, vocab_size: int) -> torch.Tensor:
    # The bootstrap model intentionally has no tokenizer dependency.  The
    # production export will replace this with the frozen Nemotron tokenizer.
    return torch.tensor([[ord(char) % vocab_size for char in text]], dtype=torch.long)


def _image_tensor(path: str) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="KestrelVL local multimodal smoke inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tool-schema", help="JSON tool schema included as context")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tool_context = ""
    if args.tool_schema:
        tool_context = "\nAvailable tools:\n" + json.dumps(json.loads(args.tool_schema), sort_keys=True)
    prompt = args.prompt + tool_context
    config = KestrelConfig.tiny(use_vision=args.image is not None)
    device = torch.device(args.device)
    model = KestrelForCausalLM(config).to(device).eval()
    ids = _encode(prompt, config.vocab_size).to(device)
    pixels = _image_tensor(str(args.image)).to(device) if args.image else None
    with torch.inference_mode():
        tokens = model.generate(ids, max_new_tokens=args.max_new_tokens, pixel_values=pixels)
    print(json.dumps({
        "token_ids": tokens[0].tolist(),
        "prompt": prompt,
        "image": str(args.image) if args.image else None,
        "tool_schema_supplied": args.tool_schema is not None,
        "note": "Bootstrap smoke path uses tiny untrained weights; it is not a quality result.",
    }, indent=2))


if __name__ == "__main__":
    main()
