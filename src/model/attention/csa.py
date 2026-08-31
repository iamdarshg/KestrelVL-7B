"""Compressed Sparse Attention (CSA) reference branch."""

import torch
from torch import nn

from .compressor import LearnedCompressor
from .lightning_indexer import LightningIndexer


class CompressedSparseAttention(nn.Module):
    def __init__(self, heads: int, kv_heads: int, head_dim: int, ratio: int, index_dim: int, topk: int) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.ratio = heads, kv_heads, head_dim, ratio
        self.compressor = LearnedCompressor(head_dim, ratio)
        self.indexer = LightningIndexer(head_dim, heads, index_dim, topk)
        self.sink_key = nn.Parameter(torch.zeros(heads, head_dim))
        self.sink_value = nn.Parameter(torch.zeros(heads, head_dim))
        self.sink_logit = nn.Parameter(torch.zeros(heads))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        # K/V are [B, KV, T, D]; query is [B, H, Q, D].
        ck, cv = self.compressor(key, key_positions, value=value)
        if ck.shape[2] == 0:
            return self.sink_value.view(1, self.heads, 1, self.head_dim).expand(
                query.shape[0], -1, query.shape[2], -1
            )
        # Retrieval uses a shared compressed key so the index remains one
        # compact structure.  Values retain KV-head identity: with two KV
        # heads each query head gathers its corresponding compressed value
        # stream.  This makes the intermediate KV-head ablation meaningful
        # instead of silently discarding every stream except stream zero.
        retrieval_key = ck.mean(dim=1)
        indices, weights, selected_valid = self.indexer(
            query, retrieval_key, query_positions, self._compressed_positions(key_positions)
        )
        query_to_kv = torch.arange(self.heads, device=query.device) * self.kv_heads // self.heads
        selected_values = cv.index_select(1, query_to_kv)
        expanded_v = selected_values[:, :, None, :, :].expand(query.shape[0], self.heads, query.shape[2], -1, -1)
        gathered = expanded_v.gather(3, indices[..., None].expand(-1, -1, -1, -1, self.head_dim))
        out = (gathered * weights[..., None]).sum(dim=3)
        has_valid = selected_valid.any(dim=-1, keepdim=False)
        sink = self.sink_value[None, :, None, :].expand_as(out)
        return torch.where(has_valid[..., None], out, sink)

    def _compressed_positions(self, positions: torch.Tensor) -> torch.Tensor:
        groups = torch.unique_consecutive(positions[0] // self.ratio)
        complete = []
        for g in groups.tolist():
            if (positions[0] // self.ratio == g).sum().item() == self.ratio:
                complete.append((g + 1) * self.ratio - 1)
        return positions.new_tensor(complete)
