"""SVD-initializable grouped low-rank output projection."""

import torch
from torch import nn


class GroupedLowRankOutput(nn.Module):
    def __init__(self, heads: int, head_dim: int, groups: int, rank: int, out_dim: int) -> None:
        super().__init__()
        if heads % groups:
            raise ValueError("groups must divide attention head count")
        self.heads, self.head_dim, self.groups = heads, head_dim, groups
        self.heads_per_group = heads // groups
        self.rank = rank
        self.out_dim = out_dim
        group_dim = self.heads_per_group * head_dim
        self.down = nn.Parameter(torch.empty(groups, group_dim, rank))
        self.up = nn.Parameter(torch.empty(groups, rank, group_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, d = x.shape
        grouped = x.reshape(b, t, self.groups, self.heads_per_group * d)
        low = torch.einsum("btgi,gir->btgr", grouped, self.down)
        reconstructed = torch.einsum("btgr,gro->btgo", low, self.up)
        return reconstructed.reshape(b, t, h * d)[..., : self.out_dim] + self.bias
