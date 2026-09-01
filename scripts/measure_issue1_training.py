"""Measure issue-#1 long-context forward/backward/step without false claims."""

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
from training.muon import build_muon_optimizer
from training.precision import optimizer_telemetry, validate_precision_policy, PrecisionPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=str(ROOT / "data/raw/nemotron"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="8192,32768,1048576")
    parser.add_argument("--mode", choices=["full_recompute", "stateful_truncated"], default="stateful_truncated")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--detach-interval", type=int, default=8192)
    parser.add_argument("--max-vram-gib", type=float, default=96.0)
    parser.add_argument("--output", default="reports/runtime/issue1_training.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("real-Nemotron training smoke requires CUDA")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, load_info = load_real_nemotron_transplant(
        KestrelConfig(),
        model_id=args.model_id,
        device=device,
        load_in_4bit=True,
        compute_dtype=compute_dtype,
    )
    model.train()
    model.enable_gradient_checkpointing(True)
    optimizer = build_muon_optimizer(model, muon_max_matrix_dimension=4096)
    precision = validate_precision_policy(
        model,
        PrecisionPolicy(
            compute_dtype=compute_dtype,
            master_weight_dtype=compute_dtype,
            gradient_dtype=compute_dtype,
        ),
    )
    records: list[dict[str, object]] = []
    for length in [int(value) for value in args.lengths.split(",") if value.strip()]:
        record: dict[str, object] = {"length": length, "mode": args.mode, "load_info": load_info.__dict__}
        try:
            ids = torch.randint(0, model.config.vocab_size, (1, length), device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = run_chunked_forward(
                model,
                ids,
                labels=ids,
                config=LongContextConfig(
                    mode=args.mode,
                    execution_chunk_tokens=args.chunk_size,
                    detach_interval_tokens=args.detach_interval,
                    max_context_tokens=1_048_576,
                ),
                optimizer=optimizer,
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
                    "token_count": result.token_count,
                    "peak_vram_gib": peak_gib,
                    "telemetry": telemetry,
                    "precision": precision,
                    "optimizer": optimizer_telemetry(optimizer),
                }
            )
            del ids, result
        except (RuntimeError, MemoryError) as exc:
            record.update({"status": "blocked", "blocker": repr(exc), "telemetry": snapshot(device)})
            torch.cuda.empty_cache()
        records.append(record)
        print(json.dumps(record, sort_keys=True, default=str), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": vars(args),
                "records": records,
                "claim_boundary": "only a passed forward/backward/step record is training evidence",
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
