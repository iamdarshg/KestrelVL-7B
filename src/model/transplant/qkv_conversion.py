"""Weight-preserving conversions from Qwen GQA projections."""

import torch


def factor_linear_svd(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A,B with A @ B approximating `weight` using truncated SVD."""
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    rank = min(rank, s.numel())
    root = s[:rank].clamp_min(0).sqrt()
    return (u[:, :rank] * root).to(weight.dtype), (root[:, None] * vh[:rank]).to(weight.dtype)


def average_gqa_kv(weight: torch.Tensor, old_kv_heads: int, new_kv_heads: int) -> torch.Tensor:
    """Average contiguous old GQA heads into the requested shared heads."""
    if old_kv_heads % new_kv_heads:
        raise ValueError("old_kv_heads must be divisible by new_kv_heads")
    if weight.shape[0] % old_kv_heads:
        raise ValueError("projection rows must be divisible by old_kv_heads")
    per_head = weight.shape[0] // old_kv_heads
    return weight.view(old_kv_heads, per_head, -1).view(new_kv_heads, old_kv_heads // new_kv_heads, per_head, -1).mean(dim=1).reshape(-1, weight.shape[1])

