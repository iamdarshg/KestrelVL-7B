"""Muon optimizer with a deliberately tiny AdamW compatibility bucket.

Muon is used for trainable matrix parameters (new attention, compressor,
indexer, grouped output, and mHC logits). AdamW is reserved for vectors and
scalars such as sinks and residual scales. Frozen Nemotron embeddings, FFNs,
norms, and LM head never enter either optimizer bucket in the local profile.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def _zeropower_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a 2-D update in fp32."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization requires a matrix")
    original_shape = matrix.shape
    transposed = matrix.shape[0] < matrix.shape[1]
    x = matrix.float().t() if transposed else matrix.float()
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.t()
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    result = x.t() if transposed else x
    return result.reshape(original_shape)


class Muon(Optimizer):
    """Muon for matrices, AdamW only for explicitly marked vector groups."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | list[dict[str, object]],
        lr: float = 0.02,
        adamw_lr: float = 1e-5,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0 or adamw_lr <= 0:
            raise ValueError("Muon and AdamW learning rates must be positive")
        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("Muon received a non-finite gradient")
                state = self.state[parameter]
                if parameter.ndim >= 2:
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = torch.zeros_like(parameter, dtype=torch.float32)
                        state["momentum_buffer"] = momentum_buffer
                    momentum_buffer.mul_(group["momentum"]).add_(grad.float(), alpha=1.0 - group["momentum"])
                    update = _zeropower_newton_schulz(momentum_buffer, group["ns_steps"])
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                else:
                    # This is intentionally the only AdamW path.
                    step = state.get("step", 0) + 1
                    state["step"] = step
                    first = state.get("exp_avg")
                    second = state.get("exp_avg_sq")
                    if first is None:
                        first = torch.zeros_like(parameter, dtype=torch.float32)
                        second = torch.zeros_like(parameter, dtype=torch.float32)
                        state["exp_avg"], state["exp_avg_sq"] = first, second
                    first.mul_(beta1).add_(grad.float(), alpha=1.0 - beta1)
                    second.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1.0 - beta2)
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    denominator = (second.sqrt() / bias_correction2**0.5).add_(group["eps"])
                    update = (first / bias_correction1) / denominator
                    lr = group["adamw_lr"]
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - lr * group["weight_decay"])
                    parameter.add_(update.to(parameter.dtype), alpha=-lr)
        return loss


def build_muon_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-5,
    weight_decay: float = 0.01,
) -> Muon:
    """Build Muon/AdamW groups and fail if there are no trainable parameters."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    optimizer = Muon(
        [
            {"params": [p for p in trainable if p.ndim >= 2], "name": "muon_matrices"},
            {"params": [p for p in trainable if p.ndim < 2], "name": "minimal_adamw_vectors"},
        ],
        lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
    )
    optimizer.matrix_parameter_count = sum(p.numel() for p in trainable if p.ndim >= 2)  # type: ignore[attr-defined]
    optimizer.vector_parameter_count = sum(p.numel() for p in trainable if p.ndim < 2)  # type: ignore[attr-defined]
    return optimizer
