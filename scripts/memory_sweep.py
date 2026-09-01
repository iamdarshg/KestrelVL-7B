"""Bounded-memory inference sweep for the declared context targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from hardware.telemetry import snapshot
from model import KestrelConfig, KestrelForCausalLM
from training.long_context import LongContextConfig, run_chunked_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lengths", default="512,2048,8192,32768,131072,262144,524288,1048576")
    parser.add_argument("--output", default="reports/runtime/memory_sweep.json")
    parser.add_argument("--max-vram-gib", type=float, default=7.5)
    args = parser.parse_args()
    device = torch.device(args.device)
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    records: list[dict[str, object]] = []
    for length in lengths:
        record: dict[str, object] = {"length": length, "device": str(device)}
        try:
            config = KestrelConfig.tiny(
                use_vision=False,
                max_position_embeddings=max(4096, length),
                sliding_window=8,
                attention_query_block=128,
            )
            model = KestrelForCausalLM(config).to(device).eval()
            ids = torch.randint(0, config.vocab_size, (1, length), device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.inference_mode():
                result = run_chunked_forward(
                    model,
                    ids,
                    config=LongContextConfig(
                        mode="stateful_truncated",
                        execution_chunk_tokens=8192,
                        detach_interval_tokens=8192,
                        max_context_tokens=max(1_048_576, length),
                    ),
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            telemetry = snapshot(device)
            peak_gib = float(telemetry.get("peak_bytes", 0)) / 2**30
            record.update(
                {
                    "status": "passed" if peak_gib <= args.max_vram_gib else "vram_overage",
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": length / max(elapsed, 1e-9),
                    "chunks": result.chunks,
                    "telemetry": telemetry,
                    "peak_vram_gib": peak_gib,
                }
            )
            del model, ids, result
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except (RuntimeError, MemoryError) as exc:
            record.update({"status": "blocked", "blocker": repr(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        records.append(record)
        print(json.dumps(record, sort_keys=True, default=str), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": str(device), "records": records}, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
