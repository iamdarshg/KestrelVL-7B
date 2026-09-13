"""Issue #7: Single-Pass mHC — fused single-mixing-pass backend (opt-in).

The existing per-layer residual path applies two sequential
:class:`~model.attention.mhc.ManifoldHyperConnection` mixings per decoder
layer (one around attention, one around the MLP). This module fuses those two
mixings into a SINGLE mixing pass with closed-form coefficients, derived
exactly from the two doubly-stochastic Sinkhorn matrices:

Sequential (old)::

    x1  = base + sa * (Ma @ [base, attn])[0]
    out = x1   + sm * (Mm @ [x1,   mlp ])[0]

Fused (this module)::

    out = alpha * base + beta * attn + gamma * mlp

with ``alpha = (1 + sm*Mm00) * (1 + sa*Ma00)``,
``beta = (1 + sm*Mm00) * (sa*Ma01)`` and ``gamma = sm*Mm01``.

Properties:

* Mathematically identical to the sequential pair given the same logits —
  promotion is evidence-gated via exact-agreement tests, never by default.
* One mixing einsum over the token stream instead of two; the
  attention-mixed state is still materialised once because the MLP input
  depends on it (compute it with :func:`mixed_attn_state`).
* No new unbounded per-token state: only scalar coefficients plus the two
  ``[S, S]`` logit matrices (same count as the replaced pair).
* ``enabled=False`` reproduces the plain residual sum
  ``base + attn + mlp``.
* ``from_sequential`` copies logits/scales from an existing pair, giving
  bit-exact agreement for A/B comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .mhc import ManifoldHyperConnection, sinkhorn

__all__ = [
    "MHC_BACKENDS",
    "SinglePassMHC",
    "mixed_attn_state",
    "build_single_pass_from_pair",
    "resolve_mhc_backend",
]

MHC_BACKENDS = ("residual", "single_pass")
_DEFAULT_BACKEND = "residual"


def resolve_mhc_backend(value: Any) -> str:
    """Validate an mHC backend name; default (and only default) is ``residual``."""
    name = str(value)
    if name not in MHC_BACKENDS:
        raise ValueError(f"unknown mhc_backend {value!r}; expected one of {MHC_BACKENDS}")
    return name


def mixed_attn_state(
    base: torch.Tensor,
    attn_update: torch.Tensor,
    mixer: ManifoldHyperConnection,
) -> torch.Tensor:
    """First-stage mixed state ``x1`` (still required as the MLP input)."""
    return mixer(base, attn_update)


def _logit_init(streams: int) -> torch.Tensor:
    """Cyclic-shift init identical to :class:`ManifoldHyperConnection`."""
    logits = torch.full((streams, streams), -2.0)
    if streams >= 2:
        logits.fill_diagonal_(-2.0)
        for i in range(streams):
            logits[i, (i + 1) % streams] = 2.0
    else:
        logits.zero_()
    return logits


class SinglePassMHC(nn.Module):
    """Fused single-mixing-pass replacement for one sequential mHC pair."""

    def __init__(
        self,
        streams: int = 2,
        sinkhorn_iters: int = 6,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if streams < 1:
            raise ValueError(f"streams must be >= 1, got {streams!r}")
        if sinkhorn_iters < 1:
            raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters!r}")
        self.streams = int(streams)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.enabled = bool(enabled)
        self.logits_attn = nn.Parameter(_logit_init(self.streams))
        self.logits_mlp = nn.Parameter(_logit_init(self.streams))
        self.residual_scale_attn = nn.Parameter(torch.ones(()))
        self.residual_scale_mlp = nn.Parameter(torch.ones(()))

    @classmethod
    def from_sequential(
        cls,
        attn_mhc: ManifoldHyperConnection,
        mlp_mhc: ManifoldHyperConnection,
    ) -> "SinglePassMHC":
        """Build a fused module that exactly reproduces ``(attn_mhc, mlp_mhc)``."""
        if attn_mhc.streams != mlp_mhc.streams:
            raise ValueError("stream counts must match for fusion")
        fused = cls(
            streams=attn_mhc.streams,
            sinkhorn_iters=attn_mhc.sinkhorn_iters,
            enabled=bool(attn_mhc.enabled and mlp_mhc.enabled),
        )
        with torch.no_grad():
            fused.logits_attn.copy_(attn_mhc.logits.detach())
            fused.logits_mlp.copy_(mlp_mhc.logits.detach())
            fused.residual_scale_attn.copy_(attn_mhc.residual_scale.detach())
            fused.residual_scale_mlp.copy_(mlp_mhc.residual_scale.detach())
        return fused

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Ma, Mm)`` doubly-stochastic mixing matrices."""
        return (
            sinkhorn(self.logits_attn, self.sinkhorn_iters),
            sinkhorn(self.logits_mlp, self.sinkhorn_iters),
        )

    def coefficients(self) -> Dict[str, torch.Tensor]:
        """Closed-form ``(alpha, beta, gamma)`` scalars for the fused pass."""
        ma, mm = self.matrices()
        work = torch.float32
        sa = self.residual_scale_attn.to(work)
        sm = self.residual_scale_mlp.to(work)
        ma = ma.to(work)
        mm = mm.to(work)
        if self.streams == 1:
            # Single stream mixes to the scalar 1.0: out = (1+sm)(1+sa)*base.
            one = torch.ones((), dtype=work, device=ma.device)
            alpha = (one + sm * mm[0, 0]) * (one + sa * ma[0, 0])
            zero = torch.zeros((), dtype=work, device=ma.device)
            return {"alpha": alpha, "beta": zero, "gamma": zero}
        alpha = (1.0 + sm * mm[0, 0]) * (1.0 + sa * ma[0, 0])
        beta = (1.0 + sm * mm[0, 0]) * (sa * ma[0, 1])
        gamma = sm * mm[0, 1]
        return {"alpha": alpha, "beta": beta, "gamma": gamma}

    def forward(
        self,
        base: torch.Tensor,
        attn_update: torch.Tensor,
        mlp_update: torch.Tensor,
    ) -> torch.Tensor:
        """Fused ``alpha*base + beta*attn + gamma*mlp`` in one mixing pass."""
        if base.shape != attn_update.shape or base.shape != mlp_update.shape:
            raise ValueError("base/attn_update/mlp_update shapes must match")
        if not self.enabled:
            return base + attn_update + mlp_update
        work_dtype = (
            torch.float32
            if base.dtype in (torch.float16, torch.bfloat16)
            else base.dtype
        )
        coef = self.coefficients()
        alpha = coef["alpha"].to(work_dtype)
        beta = coef["beta"].to(work_dtype)
        gamma = coef["gamma"].to(work_dtype)
        out = (
            alpha * base.to(work_dtype)
            + beta * attn_update.to(work_dtype)
            + gamma * mlp_update.to(work_dtype)
        )
        return out.to(dtype=base.dtype)

    # -- deterministic checkpoint/resume ----------------------------------
    def state_dict_snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "single_pass",
            "streams": int(self.streams),
            "sinkhorn_iters": int(self.sinkhorn_iters),
            "enabled": bool(self.enabled),
        }

    def load_snapshot_strict(
        self, state: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> None:
        expected = self.metadata()
        for key in ("backend", "streams", "sinkhorn_iters"):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"snapshot metadata mismatch for {key!r}: got "
                    f"{metadata.get(key)!r}, expected {expected[key]!r}"
                )
        self.load_state_dict(dict(state), strict=True)


def build_single_pass_from_pair(
    attn_mhc: ManifoldHyperConnection,
    mlp_mhc: ManifoldHyperConnection,
) -> SinglePassMHC:
    """Convenience alias for :meth:`SinglePassMHC.from_sequential`."""
    return SinglePassMHC.from_sequential(attn_mhc, mlp_mhc)
