"""Partial rotary position embedding with a YaRN-compatible frequency scale."""

import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class PartialRotaryEmbedding:
    def __init__(self, rotary_dim: int, theta: float, max_position: int, yarn_factor: float = 1.0) -> None:
        self.rotary_dim = rotary_dim
        self.theta = theta
        self.max_position = max_position
        self.yarn_factor = yarn_factor
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        max_pos = int(positions.max().item()) + 1 if positions.numel() else 1
        cached = self._cache.get(key)
        if cached is None or cached[0].shape[0] < max_pos:
            inv = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32) / self.rotary_dim))
            pos = torch.arange(max_pos, device=device, dtype=torch.float32)
            if self.yarn_factor > 1:
                pos = pos / self.yarn_factor
            freqs = torch.outer(pos, inv)
            emb = torch.cat((freqs, freqs), dim=-1)
            cached = (emb.cos().to(dtype), emb.sin().to(dtype))
            self._cache[key] = cached
        cos, sin = cached
        return cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1), sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, -1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim], positions: [batch, sequence]
        if self.rotary_dim == 0:
            return x
        cos, sin = self._cos_sin(positions, x.device, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        head, tail = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        return torch.cat((head * cos + rotate_half(head) * sin, tail), dim=-1)

