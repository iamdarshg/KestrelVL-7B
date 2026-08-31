"""Run all requested architecture ablations on the local device.

This is a bounded *architecture/recovery proxy*, not a benchmark claim. It
fits a tiny model to a fixed dense-attention teacher for a few steps, checks
prefill/decode cache equivalence, and records timing/memory. The measured
selection is only used to choose the next architecture for real recovery.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hardware.telemetry import append_jsonl
from model import KestrelConfig, KestrelForCausalLM
from model.attention.cache import KestrelCache


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(kv_heads: int, hca_ratio: int, topk: int, mhc: bool) -> KestrelConfig:
    return KestrelConfig.tiny(
        num_key_value_heads=kv_heads,
        hca_compression_ratio=hca_ratio,
        index_topk=topk,
        mhc_enabled=mhc,
        use_vision=False,
        max_position_embeddings=4096,
    )


@torch.no_grad()
def cache_error(model: KestrelForCausalLM, ids: torch.Tensor) -> float:
    full = model(ids).logits
    cache = KestrelCache()
    pieces = [model(ids[:, :1], past_key_values=cache).logits]
    for index in range(1, ids.shape[1]):
        pieces.append(model(ids[:, index : index + 1], past_key_values=cache).logits)
    decoded = torch.cat(pieces, dim=1)
    return float((full - decoded).abs().max().cpu())


def one_run(args: argparse.Namespace, device: torch.device, teacher_logits: torch.Tensor, ids: torch.Tensor, combo: tuple[int, int, int, bool]) -> dict[str, object]:
    kv_heads, hca_ratio, topk, mhc = combo
    seed_all(args.seed)
    config = model_config(*combo)
    model = KestrelForCausalLM(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    losses: list[float] = []
    for _ in range(args.steps):
        output = model(ids)
        loss = torch.nn.functional.mse_loss(output.logits.float(), teacher_logits.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    error = cache_error(model, ids[:, : args.cache_length])
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    effective_csa = min(topk, args.seq_length // config.csa_compression_ratio)
    result = {
        "kv_heads": kv_heads,
        "hca_ratio": hca_ratio,
        "index_topk": topk,
        "effective_csa_topk": effective_csa,
        "mhc_enabled": mhc,
        "steps": args.steps,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_delta": losses[0] - losses[-1],
        "cache_max_abs_error": error,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (args.seq_length * args.steps) / max(elapsed, 1e-9),
        "peak_vram_gib": peak,
        "parameters": model.parameter_count(),
        "status": "pass" if np.isfinite(losses[-1]) and error < args.cache_tolerance else "fail",
    }
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def choose(results: list[dict[str, object]]) -> dict[str, object]:
    valid = [r for r in results if r["status"] == "pass"]
    if not valid:
        raise RuntimeError("all local ablations failed correctness/finite gates")
    # Prefer representation recovery, then throughput, with a mild memory
    # penalty. This is intentionally transparent and not a public benchmark.
    return min(valid, key=lambda r: (float(r["loss_final"]), -float(r["tokens_per_second"]), float(r["peak_vram_gib"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--cache-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache-tolerance", type=float, default=5e-5)
    parser.add_argument("--output", default="reports/ablations/local_ablation_results.json")
    args = parser.parse_args()
    if args.cache_length > args.seq_length:
        raise ValueError("cache-length cannot exceed seq-length")
    device = torch.device(args.device)
    seed_all(args.seed)
    ids = torch.randint(0, 257, (1, args.seq_length), device=device)
    teacher_config = KestrelConfig.tiny(use_vision=False, layer_schedule=["sliding"] * 4, sliding_window=max(256, args.seq_length))
    teacher = KestrelForCausalLM(teacher_config).to(device).eval()
    with torch.no_grad():
        teacher_logits = teacher(ids).logits.detach()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    combos = list(itertools.product((1, 2), (64, 128), (64, 128, 256, 512), (False, True)))
    results = []
    for index, combo in enumerate(combos, 1):
        print(f"[{index}/{len(combos)}] kv={combo[0]} hca={combo[1]} topk={combo[2]} mhc={combo[3]}", flush=True)
        result = one_run(args, device, teacher_logits, ids, combo)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    winner = choose(results)
    payload = {"protocol": vars(args), "device": str(device), "ablation_count": len(results), "results": results, "winner": winner}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    final = Path("reports/ablations/final_architecture.json")
    final.write_text(json.dumps({"source":"local_ablation_results.json", "selection_policy":"lowest finite post-fit distillation proxy loss, then throughput, then peak VRAM", "selected": winner}, indent=2) + "\n", encoding="utf-8")
    print("WINNER=" + json.dumps(winner, sort_keys=True))


if __name__ == "__main__":
    main()
