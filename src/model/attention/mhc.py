"""Manifold-constrained hyperconnection (mHC) residual mixer.

The learnable connection is projected onto the doubly-stochastic manifold by
Sinkhorn normalization. A two-stream identity/permutation initialization makes
the disabled-equivalent path `base + update`, while training can learn stable
cross-stream routing without unconstrained residual amplification.
"""

import torch
from torch import nn


def sinkhorn(logits: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    x = logits.float()
    for _ in range(iterations):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
    return x.exp().to(logits.dtype)


class ManifoldHyperConnection(nn.Module):
    def __init__(self, streams: int = 2, sinkhorn_iters: int = 6, enabled: bool = True) -> None:
        super().__init__()
        self.streams = streams
        self.sinkhorn_iters = sinkhorn_iters
        self.enabled = enabled
        logits = torch.full((streams, streams), -2.0)
        if streams >= 2:
            logits.fill_diagonal_(-2.0)
            for i in range(streams):
                logits[i, (i + 1) % streams] = 2.0
        else:
            logits.zero_()
        self.logits = nn.Parameter(logits)
        self.residual_scale = nn.Parameter(torch.ones(()))

    def matrix(self) -> torch.Tensor:
        return sinkhorn(self.logits, self.sinkhorn_iters)

    def forward(self, base: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return base + update
        # First stream is the base and second is the update. Extra streams are
        # zero-filled, which keeps the API extensible without hidden copies.
        # Keep the connection and residual accumulation in FP32.  In FP16,
        # even finite residual terms can overflow during the stream einsum or
        # final addition before the next RMSNorm has a chance to rescale them.
        work_dtype = torch.float32 if base.dtype in (torch.float16, torch.bfloat16) else base.dtype
        base_work = base.to(work_dtype)
        update_work = update.to(work_dtype)
        streams = torch.zeros(
            *base.shape[:-1], self.streams, base.shape[-1], device=base.device, dtype=work_dtype
        )
        streams[..., 0, :] = base_work
        if self.streams > 1:
            streams[..., 1, :] = update_work
        mixed = torch.einsum("ij,btjd->btid", self.matrix().to(work_dtype), streams)
        output = base_work + self.residual_scale.to(work_dtype) * mixed[..., 0, :]
        return output.to(dtype=base.dtype)
