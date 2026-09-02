"""Muon optimizer with a deliberately tiny AdamW compatibility bucket.

Muon is used for trainable matrix parameters (new attention, compressor,
indexer, grouped output, and mHC logits). AdamW is reserved for vectors and
scalars such as sinks and residual scales. Oversized matrices can explicitly
use the AdamW fallback because Newton--Schulz would otherwise allocate an
unusable square Gram matrix (for example a BERT vocabulary embedding).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def _zeropower_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix-shaped update in fp32.

    Grouped projections naturally have a leading group axis, so Muon treats
    every tensor with at least two dimensions as one flattened matrix and
    restores its parameter shape afterward.
    """
    if matrix.ndim < 2:
        raise ValueError("Muon orthogonalization requires at least two dimensions")
    original_shape = matrix.shape
    # For ordinary 2-D weights this is the usual Muon update.  For grouped
    # projections (for example [groups, in_features, rank]), apply the
    # orthogonalization independently to each leading slice.  Flattening all
    # slices into one matrix would create a needlessly enormous Gram matrix.
    batched = matrix.ndim > 2
    if batched:
        flat = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1]).float()
        transposed = flat.shape[-2] < flat.shape[-1]
        x = flat.transpose(-1, -2) if transposed else flat
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    else:
        flat = matrix
        transposed = flat.shape[0] < flat.shape[1]
        x = flat.float().t() if transposed else flat.float()
        x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        gram2 = gram @ gram
        x = a * x + (b * gram + c * gram2) @ x
    result = x.transpose(-1, -2) if transposed else x
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
        momentum_dtype: torch.dtype = torch.bfloat16,
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
            momentum_dtype=momentum_dtype,
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
                use_muon = bool(group.get("use_muon", parameter.ndim >= 2))
                if parameter.ndim >= 2 and use_muon:
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = torch.zeros_like(parameter, dtype=group["momentum_dtype"])
                        state["momentum_buffer"] = momentum_buffer
                    momentum_buffer.mul_(group["momentum"]).add_(grad.to(momentum_buffer.dtype), alpha=1.0 - group["momentum"])
                    update = _zeropower_newton_schulz(momentum_buffer, group["ns_steps"])
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                else:
                    # This is intentionally the only AdamW path.  It covers
                    # vectors/scalars and explicitly excluded oversized
                    # matrices; ordinary matrices stay on Muon above.
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
    muon_max_matrix_dimension: int | None = None,
    muon_ns_steps: int = 5,
) -> Muon:
    """Build Muon/AdamW groups and fail if there are no trainable parameters.

    ``muon_max_matrix_dimension`` protects the local screen from square
    Newton--Schulz allocations on vocabulary-sized matrices.  Excluded
    matrices are placed in the explicitly named AdamW fallback group; the
    default ``None`` preserves the original all-matrix Muon behavior.
    ``muon_ns_steps`` is configurable because the Newton--Schulz refinement
    is the dominant optimizer cost on older GPUs; one step is a useful
    screening mode while the default five-step update remains the quality
    profile.
    """
    if muon_ns_steps < 1:
        raise ValueError("muon_ns_steps must be positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    muon_matrices = [
        parameter
        for parameter in trainable
        if parameter.ndim >= 2
        and (
            muon_max_matrix_dimension is None
            or max(parameter.shape) <= muon_max_matrix_dimension
        )
    ]
    muon_ids = {id(parameter) for parameter in muon_matrices}
    adamw_parameters = [parameter for parameter in trainable if id(parameter) not in muon_ids]
    optimizer = Muon(
        [
            {"params": muon_matrices, "name": "muon_matrices", "use_muon": True},
            {
                "params": adamw_parameters,
                "name": "minimal_adamw_vectors_and_oversized_matrices",
                "use_muon": False,
            },
        ],
        lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        ns_steps=muon_ns_steps,
        momentum_dtype=torch.bfloat16,
    )
    optimizer.matrix_parameter_count = sum(p.numel() for p in muon_matrices)  # type: ignore[attr-defined]
    optimizer.vector_parameter_count = sum(p.numel() for p in trainable if p.ndim < 2)  # type: ignore[attr-defined]
    optimizer.adamw_fallback_matrix_parameter_count = sum(  # type: ignore[attr-defined]
        p.numel() for p in adamw_parameters if p.ndim >= 2
    )
    optimizer.muon_parameter_count = sum(p.numel() for p in muon_matrices)  # type: ignore[attr-defined]
    optimizer.adamw_parameter_count = sum(p.numel() for p in adamw_parameters)  # type: ignore[attr-defined]
    return optimizer
