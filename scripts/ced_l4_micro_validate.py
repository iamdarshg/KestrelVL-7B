"""Batch-3 micro GPU validation for #5/#6/#7 numerics on REAL activations.

Device-agnostic and seeded. Two modes:

* ``smoke-cpu``: synthetic tensors only — proves the check logic on CPU
  with no weights, no GPU, no network. Run locally before any launch.
* ``real``: loads the pinned 9B decoder (bf16, CUDA), tiles REAL hidden
  states into S=4096 memory with a planted far span, then runs the three
  checks. Unattended; any failure raises loudly with partial evidence.

Checks: (a) Single-Pass mHC vs sequential agreement in bf16;
(b) CSA2 stack forward+backward over S=4096 with pool-bounded telemetry;
(c) hierarchical indexer far-span retrieval + recall vs dense reference.

Usage (local):  python scripts/ced_l4_micro_validate.py --mode smoke-cpu
Usage (VM):     python scripts/ced_l4_micro_validate.py --mode real --out /var/log/micro-results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

DECODER_ID = "Qwen/Qwen3.5-9B"
DECODER_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PROMPT = "The capital of France is"

# PASS tolerances: mhc bf16 tol covers pure rounding-order difference between
# one fused weighted sum and two sequential mixes (CPU bf16 calibration:
# max_abs 0.0156 ≈ 2 ulp at values O(1); tol 0.05 keeps 3× margin).
# Recall thresholds from planted-span synthetic (recall 1.0 expected).
MHC_BF16_TOL = 5e-2
SPAN_RECALL_MIN = 0.9
TOPK_RECALL_MIN = 0.9


def _resolve_device(name: str):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _peak_gb(device) -> float:
    import torch

    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    return 0.0


def check_mhc_agreement(h, dtype_label: str) -> dict:
    """Fused Single-Pass vs sequential pair on [B,T,D] hidden states."""
    import torch

    from model.attention.mhc import ManifoldHyperConnection
    from model.attention.mhc_singlepass import SinglePassMHC

    torch.manual_seed(0)
    attn = torch.randn_like(h) * 0.1
    mlp = torch.randn_like(h) * 0.1
    a = ManifoldHyperConnection(streams=2, sinkhorn_iters=6, enabled=True).to(
        device=h.device, dtype=h.dtype
    )
    m = ManifoldHyperConnection(streams=2, sinkhorn_iters=6, enabled=True).to(
        device=h.device, dtype=h.dtype
    )
    fused = SinglePassMHC.from_sequential(a, m)
    with torch.no_grad():
        expected = m(a(h, attn), mlp)
        got = fused(h, attn, mlp)
    abs_diff = (got.float() - expected.float()).abs()
    max_abs = float(abs_diff.max().item())
    denom = float(expected.float().abs().max().item())
    passed = max_abs <= MHC_BF16_TOL
    return {
        "dtype": dtype_label,
        "shape": list(h.shape),
        "max_abs_diff": max_abs,
        "rel_diff": max_abs / denom if denom > 0 else 0.0,
        "tol": MHC_BF16_TOL,
        "pass": bool(passed),
    }


def check_csa2_long(mem_k, mem_v, query, head_dim: int, top_k: int, pool: int) -> dict:
    """CSA2 stack over S positions: forward+backward, pool-bounded cost.

    ``out_proj`` is zero at init by design (identity-preserving), which
    blocks upstream grads until it moves. The check therefore opens the
    post-stage-1 regime explicitly (out_proj = 0.05, as in the CPU combined
    test) and then requires grads everywhere the forward path uses them.
    Reuse layers legitimately leave q/k untouched (carried projections are
    unused on the reuse path — known wart, see handoff).
    """
    import torch

    from model.csa2 import CSA2Config, CSA2Stack

    t0 = time.time()
    config = CSA2Config(
        top_k=top_k,
        candidate_pool_size=pool,
        num_decoder_layers=4,
        head_dim=head_dim,
        cadence=["full", "reuse", "reuse", "reindex"],
    )
    stack = CSA2Stack(config, hidden_dim=query.shape[-1],
                      memory_dim=mem_k.shape[-1]).to(device=query.device)
    if query.dtype != torch.float32:
        stack = stack.to(dtype=torch.float32)
        mem_k, mem_v, query = mem_k.float(), mem_v.float(), query.float()
    with torch.no_grad():
        for layer in stack.layers:
            layer.out_proj.weight.fill_(0.05)
    for p in stack.parameters():
        p.requires_grad_(True)
    stream, _, telemetry = stack(query, mem_k, mem_v, chunk=512)
    loss = stream.square().mean()
    loss.backward()
    per_layer_ok = []
    for layer, tel in zip(stack.layers, telemetry):
        used = ("q_proj", "k_proj", "out_proj") if tel["mode"] != "reuse" else ("out_proj",)
        ok = True
        for name in used:
            g = dict(layer.named_parameters())[name + ".weight"].grad
            if g is None or not bool(torch.isfinite(g).all()):
                ok = False
        per_layer_ok.append(ok)
    modes = [t["mode"] for t in telemetry]
    scored = [t["positions_scored"] for t in telemetry]
    grad_ok = all(per_layer_ok)
    passed = (
        bool(torch.isfinite(stream).all())
        and grad_ok
        and modes == ["full", "reuse", "reuse", "reindex"]
        and scored[1] == 0
        and scored[2] == 0
        and scored[0] == mem_k.shape[1]
        and scored[3] == pool
    )
    return {
        "seq_len": mem_k.shape[1],
        "modes": modes,
        "positions_scored": scored,
        "finite": bool(torch.isfinite(stream).all()),
        "grad_ok": bool(grad_ok),
        "per_layer_grad_ok": [bool(v) for v in per_layer_ok],
        "wall_s": round(time.time() - t0, 2),
        "pass": bool(passed),
    }


def check_indexer_span(mem_k, mem_v, query, span: tuple, top_k: int, pool: int) -> dict:
    """Planted far-span retrieval + recall vs explicit dense reference.

    Coarse projectors are aligned to identity (coarse_dim == dim) for this
    machinery check: with random projections, coarse retrieval is
    chance-level by construction, and learned-projection quality is a
    stage-2 TRAINING gate, not an architecture gate. Identity alignment
    makes coarse scores exact, so recall 1.0 validates the chunked
    pool→fine plumbing, determinism, and telemetry end to end.
    """
    import torch

    from model.sparse_indexer import (
        IndexerConfig,
        dense_reference_topk,
        recall_at_k,
        span_recall,
    )
    from model.sparse_indexer import HierarchicalIndexer

    t0 = time.time()
    dim = mem_k.shape[-1]
    indexer = HierarchicalIndexer(
        mem_dim=dim,
        config=IndexerConfig(
            candidate_pool_size=pool, top_k=top_k, chunk_size=512,
            index_dtype="bfloat16", coarse_dim=dim,
        ),
    ).to(device=mem_k.device)
    with torch.no_grad():
        eye = torch.eye(dim, device=mem_k.device)
        indexer.mem_projector.weight.copy_(eye)
        indexer.mem_projector.bias.zero_()
        indexer.query_projector.weight.copy_(eye)
        indexer.query_projector.bias.zero_()
    _, _, indices, telemetry = indexer.retrieve(
        query.float(), mem_k.float(), mem_v.float(), source="semantic"
    )
    _, ref = dense_reference_topk(query.float(), mem_k.float(), top_k)
    topk_recall = recall_at_k(indices.detach().cpu(), ref.detach().cpu())
    span_cov = span_recall(indices.detach().cpu(), span)
    passed = topk_recall >= TOPK_RECALL_MIN and span_cov >= SPAN_RECALL_MIN
    return {
        "seq_len": mem_k.shape[1],
        "span": [int(span[0]), int(span[1])],
        "topk_recall_vs_dense": topk_recall,
        "span_coverage": span_cov,
        "fine_scored": telemetry["positions_fine_scored"],
        "wall_s": round(time.time() - t0, 2),
        "pass": bool(passed),
    }


def run_smoke_cpu(device) -> dict:
    """Synthetic logic proof: tiny dims, planted span, CPU."""
    import torch

    torch.manual_seed(7)
    dim, seq, top_k, pool = 64, 256, 8, 32
    span = (200, 208)
    gen = torch.Generator().manual_seed(7)
    pattern = torch.randn(dim, generator=gen)
    pattern = pattern / pattern.norm().clamp_min(1e-6) * 6.0
    mem_k = torch.randn(1, seq, dim, generator=gen, device=device) * 0.5
    noise = torch.randn(span[1] - span[0], dim, generator=gen, device=device)
    mem_k[0, span[0]:span[1], :] = pattern.unsqueeze(0).to(device) + noise * 0.01
    mem_v = torch.randn(1, seq, dim, generator=gen, device=device) * 0.5
    query = pattern.view(1, 1, dim).to(device)
    h = torch.randn(1, 4, dim, device=device)
    out = {
        "mode": "smoke-cpu",
        "mhc": check_mhc_agreement(h, "float32"),
        "csa2": check_csa2_long(mem_k, mem_v, torch.randn(1, 2, dim, device=device),
                                head_dim=16, top_k=top_k, pool=pool),
        "indexer": check_indexer_span(mem_k, mem_v, query, span, top_k, pool),
    }
    out["status"] = "PASS" if all(v["pass"] for v in out.values() if isinstance(v, dict)) else "FAIL"
    return out


def run_real(device, seq: int, top_k: int, pool: int) -> dict:
    """Real 9B bf16 activations tiled to S=seq with a planted far span."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_all = time.time()
    out: dict = {"mode": "real", "gpu": torch.cuda.get_device_name(device)}
    tok = AutoTokenizer.from_pretrained(DECODER_ID, revision=DECODER_REV,
                                        trust_remote_code=False)
    if len(tok) != 248077:
        raise ValueError(f"tokenizer vocab {len(tok)} != 248077")
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_ID, revision=DECODER_REV, dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=False,
    ).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        h = model.model(ids)[0].detach()
    if not torch.isfinite(h.float()).all():
        raise ValueError("non-finite real hidden states")
    dim = h.shape[-1]
    out["hidden_dim"] = dim
    # Long memory: tile real states, plant a far-span copy of a real pattern.
    reps = (seq + h.shape[1] - 1) // h.shape[1]
    mem = h.repeat(1, reps, 1)[:, :seq, :].contiguous()
    span = (seq - 300, seq - 292)
    pattern = h[0, 2, :].detach()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(span[1] - span[0], dim, generator=gen,
                        device=device, dtype=torch.bfloat16) * 0.01
    mem[0, span[0]:span[1], :] = pattern.unsqueeze(0) + noise
    query = pattern.view(1, 1, dim)
    out["mhc"] = check_mhc_agreement(h[:, :8, :].contiguous(), "bfloat16")
    out["csa2"] = check_csa2_long(mem, mem.clone(), query.expand(-1, 4, -1).contiguous(),
                                  head_dim=256, top_k=top_k, pool=pool)
    out["indexer"] = check_indexer_span(mem, mem.clone(), query, span, top_k, pool)
    out["peak_vram_gb"] = _peak_gb(device)
    out["wall_s"] = round(time.time() - t_all, 1)
    out["status"] = "PASS" if all(v["pass"] for v in out.values() if isinstance(v, dict)) else "FAIL"
    if out["status"] != "PASS":
        raise RuntimeError(f"micro validation FAIL: {json.dumps(out)[:800]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke-cpu", "real"], default="smoke-cpu")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--pool", type=int, default=2048)
    parser.add_argument("--out", default="/var/log/micro-results.json")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    if args.mode == "smoke-cpu":
        results = run_smoke_cpu(device)
    else:
        if device.type != "cuda":
            raise RuntimeError("real mode needs CUDA")
        results = run_real(device, args.seq, args.top_k, args.pool)
    print("MICRO_RESULT_JSON=" + json.dumps(results, sort_keys=True))
    if args.mode == "real":
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        tb = traceback.format_exc(limit=15)
        print("MICRO_TRACEBACK=" + tb[-3000:])
        print("MICRO_RESULT_JSON=" + json.dumps({"status": "FAIL", "error": str(exc)[:500]}))
        sys.exit(1)
