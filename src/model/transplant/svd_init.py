"""Initialization helpers that retain the original dense attention geometry."""

import torch

from ..attention.module import V4FlashAttention
from .qkv_conversion import average_gqa_kv, factor_linear_svd


@torch.no_grad()
def initialize_attention_from_dense(
    attention: V4FlashAttention,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    old_kv_heads: int,
) -> dict[str, float]:
    """Project dense Q/K/V/O weights into the transplant, reporting errors.

    The returned reconstruction errors are stored with the checkpoint and are
    not treated as a training gate by themselves.
    """
    q_a, q_b = factor_linear_svd(wq.T, attention.config.q_lora_rank)
    attention.q_a.weight.copy_(q_a.T)
    attention.q_b.weight.copy_(q_b.T)
    wk_shared = average_gqa_kv(wk, old_kv_heads, attention.config.num_key_value_heads)
    wv_shared = average_gqa_kv(wv, old_kv_heads, attention.config.num_key_value_heads)
    attention.kv.weight[: wk_shared.shape[0]].copy_(wk_shared)
    attention.kv.weight[wk_shared.shape[0] :].copy_(wv_shared)
    dense_flat = wo
    # HF linear weights are [out, in], while the grouped output path consumes
    # row vectors as x @ down @ up.  Factor each diagonal block of wo.T so the
    # grouped transplant reconstructs the corresponding output subspace.
    attention.out.up.zero_()
    attention.out.down.zero_()
    groups = attention.config.output_groups
    out_per = dense_flat.shape[0] // groups
    in_per = dense_flat.shape[1] // groups
    reconstructed = torch.zeros_like(dense_flat)
    for g in range(groups):
        block_t = dense_flat[g * out_per : (g + 1) * out_per, g * in_per : (g + 1) * in_per].T
        down, up = factor_linear_svd(block_t, attention.config.output_rank)
        rank = down.shape[1]
        attention.out.down[g, :, :rank].copy_(down)
        attention.out.up[g, :rank, :].copy_(up)
        reconstructed[g * out_per : (g + 1) * out_per, g * in_per : (g + 1) * in_per].copy_((down @ up).T)
    return {
        "q_relative_error": _rel_error(wq.T, q_a @ q_b),
        "o_relative_error": _rel_error(dense_flat, reconstructed),
    }


def _rel_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float((reference.float() - estimate.float()).norm() / reference.float().norm().clamp_min(1e-12))
