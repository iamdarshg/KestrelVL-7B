"""InternViT-to-language adaptive projector."""

import torch
from torch import nn

from .resampler import TokenBudgetResampler


class AdaptiveVisionProjector(nn.Module):
    def __init__(self, vision_dim: int = 1024, language_dim: int = 3584, budget: int = 1024) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, language_dim * 2),
            nn.GELU(),
            nn.Linear(language_dim * 2, language_dim),
        )
        self.resampler = TokenBudgetResampler(language_dim, budget)
        self.default_budget = budget

    def forward(self, visual_tokens: torch.Tensor, budget: int | None = None) -> torch.Tensor:
        return self.resampler(self.mlp(self.norm(visual_tokens)), budget)

