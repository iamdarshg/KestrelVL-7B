"""Causal local attention without allocating a sequence-by-sequence mask."""

import torch
import torch.nn.functional as F


def sliding_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    window: int,
    query_block: int = 256,
) -> torch.Tensor:
    """Compute local causal attention in bounded query blocks.

    Shapes are `[B, H, T, D]`; the largest attention intermediate is
    `[B, H, query_block, window]`, independent of the total context length.
    """
    if window < 1:
        raise ValueError("window must be positive")
    outputs: list[torch.Tensor] = []
    q_len = query.shape[2]
    for start in range(0, q_len, query_block):
        end = min(q_len, start + query_block)
        qpos = query_positions[:, start:end]
        low = int(qpos.min().item()) - window + 1
        high = int(qpos.max().item()) + 1
        key_idx = ((key_positions[0] >= low) & (key_positions[0] < high)).nonzero(as_tuple=False).flatten()
        if key_idx.numel() == 0:
            raise RuntimeError("cache does not contain a key in the requested local window")
        ks, ke = int(key_idx[0]), int(key_idx[-1]) + 1
        k = key[:, :, ks:ke]
        v = value[:, :, ks:ke]
        allowed = (key_positions[:, ks:ke].unsqueeze(1).unsqueeze(1) <= qpos.unsqueeze(1).unsqueeze(-1))
        allowed &= key_positions[:, ks:ke].unsqueeze(1).unsqueeze(1) >= (qpos.unsqueeze(1).unsqueeze(-1) - window + 1)
        out = F.scaled_dot_product_attention(query[:, :, start:end], k, v, attn_mask=allowed, dropout_p=0.0)
        outputs.append(out)
    return torch.cat(outputs, dim=2)

