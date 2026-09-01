"""Measure the real-Nemotron issue-#1 long-context inference contract.

This script intentionally reports blockers instead of converting an OOM or an
unmeasured run into a capability claim.  It computes only the final token logit
per chunk, so the vocabulary dimension is not multiplied by context length.
"""

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
from model.configuration import KestrelConfig
from model.nemotron import load_real_nemotron_transplant
from training.long_context import LongContextConfig, run_chunked_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=str(ROOT / "data/raw/nemotron"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="512,4096,32768,131072,1048576")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-vram-gib", type=float, default=7.5)
    parser.add_argument("--output", default="reports/runtime/issue1_inference.json")
    parser.add_argument(
        "--compressed-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the CSA/HCA branch for memory stress; an initialized gate is otherwise zero",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("issue-#1 real inference measurement requires CUDA; use memory_sweep.py for CPU reference tests")
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    model, load_info = load_real_nemotron_transplant(
        # Issue #1 text-memory probe: vision is measured separately so a
        # missing InternViT artifact cannot silently change the VRAM contract.
        KestrelConfig(use_vision=False),
        model_id=args.model_id,
        device=device,
        load_in_4bit=True,
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    if args.compressed_branch:
        for layer in model.layers:
            if layer.attention.csa is not None:
                layer.attention.compressed_gate.data.fill_(1.0)
    model.eval()
    records: list[dict[str, object]] = []
    for length in lengths:
        record: dict[str, object] = {"length": length, "device": str(device), "load_info": load_info.__dict__}
        try:
            ids = torch.randint(0, model.config.vocab_size, (1, length), device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.inference_mode():
                result = run_chunked_forward(
                    model,
                    ids,
                    config=LongContextConfig(
                        mode="stateful_truncated",
                        execution_chunk_tokens=args.chunk_size,
                        detach_interval_tokens=args.chunk_size,
                        max_context_tokens=1_048_576,
                    ),
                )
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
                    "peak_vram_gib": peak_gib,
                    "telemetry": telemetry,
                    "cache_memory_bytes": result.telemetry["cache_memory_bytes"],
                    "evidence_label": result.telemetry["evidence_label"],
                    "logits_to_keep": 1,
                    "compressed_branch": args.compressed_branch,
                }
            )
            del ids, result
        except (RuntimeError, MemoryError) as exc:
            record.update({"status": "blocked", "blocker": repr(exc)})
            torch.cuda.empty_cache()
        records.append(record)
        print(json.dumps(record, sort_keys=True, default=str), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": vars(args),
                "model_id": args.model_id,
                "records": records,
                "claim_boundary": "only records with status=passed are measured inference evidence",
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
