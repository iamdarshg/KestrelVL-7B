"""Reward-normalized specialist update primitives."""

import torch


def grpo_objective(logprob: torch.Tensor, old_logprob: torch.Tensor, advantages: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor:
    ratio = (logprob - old_logprob).exp()
    clipped = ratio.clamp(1 - clip_range, 1 + clip_range)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()

