"""Issue #5: CSA2 Full / Reindex / Reuse sparse attention over global memory.

Operates on plain ``torch.Tensor`` memory (``[B, S, D]``) duck-typed from the
encoder bridge; this module never imports ``encoder_bridge`` at top level
(optional lazy import only, see :func:`_optional_encoder_bridge`).

Causal safety: memory positions are prompt-derived and fully visible to every
query. Queries are scored independently per position, so query ``t`` never
reads hidden states from positions ``> t``. Any decoder self-attention over
generated tokens (outside this module) must still apply a query-side causal
mask; this module performs memory cross-attention only and is causally safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "LayerMode",
    "CSA2Config",
    "ReuseWithoutPriorError",
    "topk_deterministic",
    "dense_reference_attention",
    "CSA2Layer",
    "CSA2Stack",
]

_VALID_MODES = ("full", "reindex", "reuse")
_INDEX_BYTES = 8  # int64 per stored index


class LayerMode(str, Enum):
    """Decoder-layer sparse-attention mode."""

    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


class ReuseWithoutPriorError(ValueError):
    """Raised when a reuse layer/stack has no published prior indices."""


def _normalize_mode(mode: Any) -> str:
    value = mode.value if isinstance(mode, LayerMode) else str(mode)
    if value not in _VALID_MODES:
        raise ValueError(f"invalid layer mode {mode!r}; expected one of {_VALID_MODES}")
    return value


@dataclass
class CSA2Config:
    """Cadence-driven sparse-attention configuration (constructor-owned)."""

    top_k: int = 2
    candidate_pool_size: int = 8
    cadence: List[str] = field(default_factory=lambda: ["full", "reuse", "reuse", "reindex"])
    num_decoder_layers: int = 4
    head_dim: int = 8
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.cadence = [_normalize_mode(m) for m in self.cadence]
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on bad modes, top_k<1, or non-positive dims."""
        if not isinstance(self.cadence, list) or not self.cadence:
            raise ValueError("cadence must be a non-empty list of modes")
        for mode in self.cadence:
            if mode not in _VALID_MODES:
                raise ValueError(f"invalid cadence entry {mode!r}")
        for name in ("top_k", "candidate_pool_size", "num_decoder_layers", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

    def layer_mode(self, i: int) -> str:
        """Map layer index to mode via cadence cyclically."""
        if isinstance(i, bool) or not isinstance(i, int) or i < 0:
            raise ValueError(f"layer index must be an int >= 0, got {i!r}")
        self.validate()
        return self.cadence[i % len(self.cadence)]


def _lex_topk(vals: Tensor, gidx: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Top-k by (score desc, global-index asc) over the last dim."""
    n = vals.shape[-1]
    k = max(1, min(int(k), n))
    order_idx = torch.argsort(gidx, dim=-1, stable=True)
    vals_s = torch.gather(vals, -1, order_idx)
    idx_s = torch.gather(gidx, -1, order_idx)
    order_score = torch.argsort(vals_s, dim=-1, descending=True, stable=True)
    take = order_score[..., :k]
    return torch.gather(vals_s, -1, take), torch.gather(idx_s, -1, take)


def topk_deterministic(scores: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Deterministic Top-K with explicit tie-breaking (score desc, index asc).

    No randomness. Returns ``(values, indices)`` sorted by score descending;
    ties resolve to the smaller memory index first.
    """
    if not isinstance(scores, Tensor):
        raise ValueError("scores must be a torch.Tensor")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an int >= 1, got {k!r}")
    n = scores.shape[-1]
    kk = min(k, n)
    base = torch.arange(n, device=scores.device).expand(scores.shape)
    vals, idx = _lex_topk(scores, base, kk)
    return vals, idx.to(torch.long)


def dense_reference_attention(hidden: Tensor, mem_k: Tensor, mem_v: Tensor) -> Tensor:
    """Reference-only exact dense softmax attention over memory (may materialize).

    Scores ``hidden @ mem_k^T / sqrt(D)`` densely as ``[B, T, S]``; this path
    exists solely for tiny-sequence correctness testing, never for production.
    Requires ``hidden.shape[-1] == mem_k.shape[-1]``.
    """
    if hidden.dim() != 3 or mem_k.dim() != 3 or mem_v.dim() != 3:
        raise ValueError("hidden/mem_k/mem_v must be 3D tensors")
    if hidden.shape[-1] != mem_k.shape[-1]:
        raise ValueError("hidden last dim must equal mem_k last dim for reference")
    if mem_k.shape[:2] != mem_v.shape[:2] or mem_k.shape[1] != mem_v.shape[1]:
        raise ValueError("mem_k/mem_v batch/seq slice must match")
    scale = 1.0 / math.sqrt(max(1, mem_k.shape[-1]))
    scores = torch.matmul(hidden, mem_k.transpose(1, 2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, mem_v)


class CSA2Layer(nn.Module):
    """One decoder layer with Full / Reindex / Reuse sparse memory attention."""

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        top_k: int = 2,
        head_dim: Optional[int] = None,
        candidate_pool_size: int = 8,
        mode: Any = "full",
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        contribution_enabled: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("memory_dim", memory_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be an int >= 1, got {top_k!r}")
        hd = hidden_dim if head_dim is None else head_dim
        if isinstance(hd, bool) or not isinstance(hd, int) or hd < 1:
            raise ValueError(f"head_dim must be an int >= 1, got {head_dim!r}")
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.top_k = int(top_k)
        self.head_dim = int(hd)
        self.candidate_pool_size = int(candidate_pool_size)
        self.mode = _normalize_mode(mode)
        self.dtype = dtype
        self.device = device
        self.contribution_enabled = bool(contribution_enabled)
        self.q_proj = nn.Linear(hidden_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(memory_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.out_proj.weight.zero_()  # identity-preserving at init
        self.to(device=device, dtype=dtype)

    def _telemetry(
        self, mode: str, positions_scored: int, pool_width: Optional[int]
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "positions_scored": int(positions_scored),
            "top_k": int(self.top_k),
            "candidate_pool_size": int(
                pool_width if pool_width is not None else self.candidate_pool_size
            ),
            "index_bytes_per_token": int(self.top_k * _INDEX_BYTES),
            "effective_top_k": int(self.top_k),
        }

    def _gather_values(self, mem_v: Tensor, indices: Tensor) -> Tensor:
        b, t, k = indices.shape
        wide = mem_v.unsqueeze(1).expand(b, t, mem_v.shape[1], mem_v.shape[2])
        return torch.gather(wide, 2, indices.unsqueeze(-1).expand(b, t, k, wide.shape[3]))

    def _weighted_output(
        self, mem_v: Tensor, indices: Tensor, sel_scores: Tensor, enabled: bool
    ) -> Tensor:
        weights = torch.softmax(sel_scores, dim=-1)
        v_sel = self._gather_values(mem_v, indices)
        attended = (weights.unsqueeze(-1) * v_sel).sum(dim=2)
        if not enabled:
            return torch.zeros(
                attended.shape[0],
                attended.shape[1],
                self.hidden_dim,
                device=attended.device,
                dtype=self.out_proj.weight.dtype,
            )
        return self.out_proj(attended)

    def forward(
        self,
        hidden: Tensor,
        mem_k: Tensor,
        mem_v: Tensor,
        prior_indices: Optional[Tensor] = None,
        candidate_pool: Optional[Tensor] = None,
        chunk: int = 4,
        materialized_for_test_only: bool = False,
        contribution_enabled: Optional[bool] = None,
        mode: Optional[Any] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Sparse memory attention for one layer.

        ``mode`` defaults to ``self.mode``. FULL ignores prior/pool inputs and
        scores all ``S`` positions in ``chunk``-sized blocks (never allocating
        ``[B, T, S]`` unless ``materialized_for_test_only``). REINDEX scores
        only ``candidate_pool`` positions. REUSE replays ``prior_indices``
        with zero scoring.
        """
        eff_mode = _normalize_mode(mode) if mode is not None else self.mode
        enabled = (
            self.contribution_enabled
            if contribution_enabled is None
            else bool(contribution_enabled)
        )
        if hidden.dim() != 3 or mem_k.dim() != 3 or mem_v.dim() != 3:
            raise ValueError("hidden/mem_k/mem_v must be [B,T,H]/[B,S,D] tensors")
        b, t, h = hidden.shape
        s, d = mem_k.shape[1], mem_k.shape[2]
        if h != self.hidden_dim or d != self.memory_dim:
            raise ValueError("hidden/memory dims disagree with layer config")
        if mem_v.shape[0] != b or mem_v.shape[1] != s:
            raise ValueError("mem_k/mem_v batch/seq must match")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("chunk must be an int >= 1")
        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        queries = self.q_proj(hidden)
        keys_all = self.k_proj(mem_k)

        if eff_mode == "reuse":
            if prior_indices is None:
                raise ReuseWithoutPriorError("reuse mode requires prior_indices")
            if prior_indices.shape[:2] != (b, t):
                raise ValueError("prior_indices must be [B,T,K] matching hidden")
            indices = prior_indices.to(torch.long).clone()
            kk = indices.shape[-1]
            uniform = torch.zeros(b, t, kk, device=hidden.device, dtype=torch.float32)
            out = self._weighted_output(mem_v, indices, uniform, enabled)
            return out, indices, self._telemetry("reuse", 0, None)

        if eff_mode == "reindex":
            if candidate_pool is None:
                raise ValueError("reindex mode requires candidate_pool [B,C]")
            if candidate_pool.dim() != 2 or candidate_pool.shape[0] != b:
                raise ValueError("candidate_pool must be [B,C]")
            pool = candidate_pool.to(torch.long)
            c = pool.shape[1]
            eff_k = min(self.top_k, c)
            # Batched pool-key gather without [B,T,S] scores.
            flat_pool = pool.unsqueeze(-1).expand(b, c, keys_all.shape[2])
            k_pool = torch.gather(keys_all, 1, flat_pool)
            if materialized_for_test_only:
                scores = torch.matmul(queries, k_pool.transpose(1, 2)) * scale
                base = torch.arange(c, device=hidden.device).expand(b * t, c)
                vals, loc = _lex_topk(scores.reshape(b * t, c), base, eff_k)
                vals = vals.reshape(b, t, eff_k)
                loc = loc.reshape(b, t, eff_k)
            else:
                run_vals: Optional[Tensor] = None
                run_loc: Optional[Tensor] = None
                for start in range(0, c, chunk):
                    stop = min(start + chunk, c)
                    sc = torch.matmul(queries, k_pool[:, start:stop, :].transpose(1, 2))
                    sc = sc * scale
                    glob = torch.arange(start, stop, device=hidden.device).expand(
                        b, t, stop - start
                    )
                    if run_vals is None:
                        cand_vals, cand_loc = sc, glob
                    else:
                        cand_vals = torch.cat([run_vals, sc], dim=-1)
                        cand_loc = torch.cat([run_loc, glob], dim=-1)
                    keep = min(eff_k, cand_vals.shape[-1])
                    run_vals, run_loc = _lex_topk(cand_vals, cand_loc, keep)
                assert run_vals is not None and run_loc is not None
                vals, loc = run_vals, run_loc
            indices = torch.gather(pool.unsqueeze(1).expand(b, t, c), 2, loc)
            out = self._weighted_output(mem_v, indices, vals, enabled)
            return out, indices, self._telemetry("reindex", c, c)

        # FULL: chunked scoring over all S; never materialize [B,T,S] by default.
        eff_k = min(self.top_k, s)
        if materialized_for_test_only:
            scores_full = torch.matmul(queries, keys_all.transpose(1, 2)) * scale
            base = torch.arange(s, device=hidden.device).expand(b * t, s)
            vals_f, idx_f = _lex_topk(scores_full.reshape(b * t, s), base, eff_k)
            indices = idx_f.reshape(b, t, eff_k)
            vals = vals_f.reshape(b, t, eff_k)
        else:
            run_vals = None
            run_idx = None
            for start in range(0, s, chunk):
                stop = min(start + chunk, s)
                sc = torch.matmul(queries, keys_all[:, start:stop, :].transpose(1, 2))
                sc = sc * scale  # [B,T,L]; only a chunk-wide slice, never [B,T,S]
                glob = torch.arange(start, stop, device=hidden.device).expand(b, t, stop - start)
                if run_vals is None:
                    cand_vals, cand_idx = sc, glob
                else:
                    cand_vals = torch.cat([run_vals, sc], dim=-1)
                    cand_idx = torch.cat([run_idx, glob], dim=-1)
                keep = min(eff_k, cand_vals.shape[-1])
                run_vals, run_idx = _lex_topk(cand_vals, cand_idx, keep)
            assert run_vals is not None and run_idx is not None
            indices, vals = run_idx, run_vals
        out = self._weighted_output(mem_v, indices, vals, enabled)
        return out, indices, self._telemetry("full", s, None)


def _pool_from_indices(indices: Tensor, pool_size: int) -> Tensor:
    """Deterministic per-batch pool: first-appearance unique ids, pad by repeat."""
    b = indices.shape[0]
    pools: List[List[int]] = []
    for bi in range(b):
        seen: List[int] = []
        known = set()
        for v in indices[bi].reshape(-1).tolist():
            iv = int(v)
            if iv not in known:
                known.add(iv)
                seen.append(iv)
        if not seen:
            seen = [0]
        while len(seen) < pool_size:
            seen.append(seen[-1])
        pools.append(seen[:pool_size])
    return torch.tensor(pools, dtype=torch.long, device=indices.device)


class CSA2Stack(nn.Module):
    """Owns per-layer modules plus cadence; threads Full/Reindex/Reuse state."""

    def __init__(
        self,
        config: CSA2Config,
        hidden_dim: int,
        memory_dim: int,
        all_full: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(config, CSA2Config):
            raise ValueError("config must be a CSA2Config")
        config.validate()
        self.config = config
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.all_full = bool(all_full)
        modes = (
            ["full"] * config.num_decoder_layers
            if all_full
            else [config.layer_mode(i) for i in range(config.num_decoder_layers)]
        )
        self.layers = nn.ModuleList(
            [
                CSA2Layer(
                    hidden_dim=hidden_dim,
                    memory_dim=memory_dim,
                    top_k=config.top_k,
                    head_dim=config.head_dim,
                    candidate_pool_size=config.candidate_pool_size,
                    mode=m,
                    dtype=config.dtype,
                    device=device,
                )
                for m in modes
            ]
        )
        self._published_indices: Optional[Tensor] = None
        self._candidate_pool: Optional[Tensor] = None

    @classmethod
    def all_full_mode(
        cls, config: CSA2Config, hidden_dim: int, memory_dim: int, device: str = "cpu"
    ) -> "CSA2Stack":
        """Migration reference: every layer runs FULL (dense-over-memory sparse)."""
        return cls(config, hidden_dim, memory_dim, all_full=True, device=device)

    def mode_metadata(self) -> List[str]:
        """Per-layer mode list."""
        return [str(layer.mode) for layer in self.layers]

    def forward(
        self,
        hidden: Tensor,
        mem_k: Tensor,
        mem_v: Tensor,
        chunk: int = 4,
        materialized_for_test_only: bool = False,
        contribution_enabled: Optional[bool] = None,
    ) -> Tuple[Tensor, List[Tensor], List[Dict[str, Any]]]:
        """Thread Full/Reindex/Reuse state residually (``h = h + contrib``)."""
        stream = hidden
        indices_per_layer: List[Tensor] = []
        telemetry: List[Dict[str, Any]] = []
        for layer in self.layers:
            assert isinstance(layer, CSA2Layer)
            if layer.mode == "full":
                out, idx, tel = layer(
                    stream,
                    mem_k,
                    mem_v,
                    chunk=chunk,
                    materialized_for_test_only=materialized_for_test_only,
                    contribution_enabled=contribution_enabled,
                )
                self._published_indices = idx.detach().clone()
                self._candidate_pool = _pool_from_indices(
                    idx.detach(), self.config.candidate_pool_size
                )
            elif layer.mode == "reindex":
                if self._candidate_pool is None:
                    raise ValueError("reindex layer has no published candidate pool")
                pool = self._candidate_pool
                if pool.shape[0] != stream.shape[0]:
                    raise ValueError("stale candidate pool batch mismatch")
                out, idx, tel = layer(
                    stream,
                    mem_k,
                    mem_v,
                    candidate_pool=pool,
                    chunk=chunk,
                    materialized_for_test_only=materialized_for_test_only,
                    contribution_enabled=contribution_enabled,
                )
                self._published_indices = idx.detach().clone()
            else:  # reuse
                if self._published_indices is None:
                    raise ReuseWithoutPriorError("reuse layer has no prior indices")
                prior = self._published_indices
                if prior.shape[0] != stream.shape[0] or prior.shape[1] != stream.shape[1]:
                    raise ReuseWithoutPriorError("stale prior indices shape mismatch")
                out, idx, tel = layer(
                    stream,
                    mem_k,
                    mem_v,
                    prior_indices=prior,
                    chunk=chunk,
                    materialized_for_test_only=materialized_for_test_only,
                    contribution_enabled=contribution_enabled,
                )
            stream = stream + out  # identity-preserving residual
            indices_per_layer.append(idx)
            telemetry.append(tel)
        return stream, indices_per_layer, telemetry

    # -- serialization ------------------------------------------------------
    def state_dict_snapshot(self) -> Dict[str, Tensor]:
        snap = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        if self._published_indices is not None:
            snap["published.indices"] = self._published_indices.detach().cpu().clone()
        if self._candidate_pool is not None:
            snap["published.pool"] = self._candidate_pool.detach().cpu().clone()
        return snap

    def metadata(self) -> Dict[str, Any]:
        return {
            "cadence": list(self.config.cadence),
            "modes": self.mode_metadata(),
            "top_k": int(self.config.top_k),
            "candidate_pool_size": int(self.config.candidate_pool_size),
            "num_decoder_layers": int(self.config.num_decoder_layers),
            "head_dim": int(self.config.head_dim),
            "hidden_dim": int(self.hidden_dim),
            "memory_dim": int(self.memory_dim),
            "dtype": str(self.config.dtype),
            "all_full": bool(self.all_full),
        }

    def load_snapshot_strict(self, snapshot: Dict[str, Tensor], metadata: Dict[str, Any]) -> None:
        expected = self.metadata()
        for key in (
            "cadence",
            "modes",
            "top_k",
            "candidate_pool_size",
            "num_decoder_layers",
            "head_dim",
            "hidden_dim",
            "memory_dim",
            "dtype",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"snapshot metadata mismatch for {key!r}: got "
                    f"{metadata.get(key)!r}, expected {expected[key]!r}"
                )
        published_idx = snapshot.pop("published.indices", None)
        published_pool = snapshot.pop("published.pool", None)
        try:
            self.load_state_dict(dict(snapshot), strict=True)
        finally:
            pass
        self._published_indices = (
            published_idx.detach().clone() if published_idx is not None else None
        )
        self._candidate_pool = (
            published_pool.detach().clone() if published_pool is not None else None
        )


def _optional_encoder_bridge() -> Dict[str, Any]:
    """Lazy, optional access to the encoder bridge (never imported at top level)."""
    import importlib

    try:
        return {"model.encoder_bridge": importlib.import_module("model.encoder_bridge")}
    except Exception:
        return {"model.encoder_bridge": None}
