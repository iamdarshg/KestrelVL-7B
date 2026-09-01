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
        compressed, compressed_values, _, _ = self.forward_with_positions(x, positions, value=value)
        return compressed, compressed_values

    def forward_with_positions(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compress complete absolute groups and report the consumed prefix.

        ``positions`` may start in the middle of a stream.  A group is emitted
        only when all ``ratio`` members are present, which makes a prefill and
        autoregressive decode produce the same compressed state.  The returned
        ``consumed`` value is a prefix length; callers keep the remainder as a
        pending raw group in the KV cache.
        """
        # x is [B, KV, T, D]. Only complete groups are emitted so a prefill
        # and token-by-token decode produce identical compressed state.
        if value is not None and value.shape != x.shape:
            raise ValueError("value must have the same shape as x")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.ndim != 2 or positions.shape[0] != x.shape[0] or positions.shape[1] != x.shape[2]:
            raise ValueError("positions must have shape [B, T]")
        value_source = value if value is not None else x
        group = positions[0].to(torch.long) // self.ratio
        unique = torch.unique_consecutive(group)
        chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        position_chunks: list[int] = []
        consumed = 0
        for g in unique.tolist():
            idx = (group == g).nonzero(as_tuple=False).flatten()
            if idx.numel() != self.ratio:
                break
            chunks.append(x.index_select(2, idx).mean(dim=2))
            value_chunks.append(value_source.index_select(2, idx).mean(dim=2))
            position_chunks.append(int((g + 1) * self.ratio - 1))
            consumed = max(consumed, int(idx[-1]) + 1)
        if not chunks:
            empty = x[:, :, :0]
            return empty, empty, positions[:, :0], 0
        compressed = torch.stack(chunks, dim=2)
        compressed_values = torch.stack(value_chunks, dim=2)
        out_positions = positions.new_tensor(position_chunks).view(1, -1).expand(x.shape[0], -1)
        return self.key_mix(compressed), self.value_mix(compressed_values), out_positions, consumed
