"""Chunked sparse retrieval indexer; no dense T-by-T mask is materialized."""

import torch
from torch import nn


class LightningIndexer(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, index_dim: int, topk: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.index_dim = index_dim
        self.topk = topk
        # Query heads already carry their head identity, so one projection is
        # shared per head. Keys are expanded per query head for retrieval.
        self.query = nn.Linear(head_dim, index_dim, bias=False)
        self.key = nn.Linear(head_dim, num_heads * index_dim, bias=False)
        self.scale = index_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # query [B,H,Q,D], key [B,M,D]
        b, h, q, _ = query.shape
        if h != self.num_heads:
            raise ValueError(f"indexer expected {self.num_heads} heads, got {h}")
        m = key.shape[1]
        # Compression naturally produces one shared position vector while
        # callers of the public indexer often provide [B, M].  Normalize both
        # forms here so the reference path has one unambiguous broadcast
        # contract and cannot accidentally construct an H=1 validity mask.
        if query_positions.ndim == 1:
            query_positions = query_positions.unsqueeze(0)
        if key_positions.ndim == 1:
            key_positions = key_positions.unsqueeze(0)
        if query_positions.shape != (b, q):
            raise ValueError(f"query_positions must have shape {(b, q)}, got {tuple(query_positions.shape)}")
        if key_positions.shape != (b, m):
            raise ValueError(f"key_positions must have shape {(b, m)}, got {tuple(key_positions.shape)}")
        qi = self.query(query)
        ki = self.key(key).view(b, m, h, self.index_dim).permute(0, 2, 1, 3)
        scores = torch.einsum("bhqd,bhmd->bhqm", qi, ki) * self.scale
        allowed = key_positions[:, None, None, :] < query_positions[:, None, :, None]
        allowed = allowed.expand(-1, h, -1, -1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        valid_any = allowed.any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid_any, scores, torch.zeros_like(scores))
        k = min(self.topk, max(1, m))
        values, indices = safe_scores.topk(k=k, dim=-1)
        selected_valid = allowed.gather(-1, indices)
        weights = torch.softmax(values.masked_fill(~selected_valid, float("-inf")), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return indices, weights, selected_valid
