"""Budget-aware visual token retention."""

import torch
from torch import nn


class TokenBudgetResampler(nn.Module):
    def __init__(self, hidden_size: int, default_budget: int = 1024) -> None:
        super().__init__()
        self.default_budget = default_budget
        self.salience = nn.Linear(hidden_size, 1)

    def forward(self, tokens: torch.Tensor, budget: int | None = None) -> torch.Tensor:
        budget = min(tokens.shape[1], budget or self.default_budget)
        if budget == tokens.shape[1]:
            return tokens
        scores = self.salience(tokens).squeeze(-1)
        indices = scores.topk(budget, dim=1).indices.sort(dim=1).values
        return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))

