"""Learned sequence compression aligned to absolute compression groups."""

import torch
from torch import nn


class LearnedCompressor(nn.Module):
    def __init__(self, head_dim: int, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.key_mix = nn.Linear(head_dim, head_dim, bias=False)
        self.value_mix = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.eye_(self.key_mix.weight)
        nn.init.eye_(self.value_mix.weight)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x is [B, KV, T, D]. Only complete groups are emitted so a prefill
        # and token-by-token decode produce identical compressed state.
        if value is not None and value.shape != x.shape:
            raise ValueError("value must have the same shape as x")
        value_source = value if value is not None else x
        group = positions[0] // self.ratio
        unique = torch.unique_consecutive(group)
        chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        out_pos: list[int] = []
        for g in unique.tolist():
            idx = (group == g).nonzero(as_tuple=False).flatten()
            if idx.numel() != self.ratio:
                continue
            chunks.append(x.index_select(2, idx).mean(dim=2))
            value_chunks.append(value_source.index_select(2, idx).mean(dim=2))
            out_pos.append(int((g + 1) * self.ratio - 1))
        if not chunks:
            empty = x[:, :, :0]
            return empty, empty
        compressed = torch.stack(chunks, dim=2)
        compressed_values = torch.stack(value_chunks, dim=2)
        return self.key_mix(compressed), self.value_mix(compressed_values)
