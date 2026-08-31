"""Measure the local developer profile and enforce its VRAM ceiling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from hardware.telemetry import assert_vram_budget, snapshot
from model import KestrelConfig, KestrelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_4060_8gb")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--output", default="reports/runtime/hardware.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = KestrelConfig.tiny(use_vision=False, max_position_embeddings=max(4096, args.sequence_length))
    model = KestrelForCausalLM(config).to(device).train()
    ids = torch.randint(0, config.vocab_size, (1, args.sequence_length), device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(ids, labels=ids)
    assert output.loss is not None
    output.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    record = {"profile": args.profile, "sequence_length": args.sequence_length, "elapsed_seconds": elapsed, "tokens_per_second": args.sequence_length / max(elapsed, 1e-9), "device": str(device), "telemetry": snapshot(device)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    if device.type == "cuda":
        assert_vram_budget(7.5, device)
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()

