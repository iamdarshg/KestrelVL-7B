"""Export a Kestrel state/config pair without pretending it is a trained release."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from model import KestrelConfig, KestrelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = KestrelConfig.tiny(use_vision=False)
    model = KestrelForCausalLM(config)
    model.load_state_dict(torch.load(source / "model.pt", map_location="cpu"))
    torch.save(model.state_dict(), output / "pytorch_model.bin")
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

