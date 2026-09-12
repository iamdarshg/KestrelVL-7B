"""Issue #4: EncoderMemoryBridge + shared encoder-derived global KV.

Minimal bridge mapping encoder hidden states to a shared global K/V memory
consumed by the decoder-side hook (owned by another agent in ``ced.py``).

Pipeline (exact): RMSNorm(encoder_dim) -> Linear(encoder_dim -> memory_dim,
bias False) -> SwiGLU-gated residual adapter (zero-init output scale) ->
K/V projections (Linear memory_dim -> memory_dim each).

No cross-attention/indexer logic lives here (issues #5/#6); this module only
exposes ``dense_reference()`` / ``sparse_memory_stub()`` for later validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .ced_config import DECODER_HIDDEN_SIZE, ENCODER_HIDDEN_SIZE

_VALID_SOURCE_TAPS = ("semantic", "detail")
_DTYPE_BY_NAME = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


def _fingerprint_tensors(
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    seq_len: int,
    memory_dim: int,
    dtype_name: str,
    device_name: str,
    source_tap: str,
) -> str:
    """Deterministic sha256 over canonical float32 bytes + metadata."""
    digest = hashlib.sha256()
    for tensor in (global_k, global_v):
        canonical = tensor.detach().to("cpu", torch.float32).contiguous()
        digest.update(canonical.numpy().tobytes())
    digest.update(f"{seq_len}|{memory_dim}|{dtype_name}|{device_name}|{source_tap}".encode())
    return digest.hexdigest()


@dataclass
class EncoderMemory:
    """Shared global K/V memory derived from encoder states."""

    global_k: torch.Tensor
    global_v: torch.Tensor
    seq_len: int
    memory_dim: int
    dtype: str = field(default=str(torch.float32))
    device: str = field(default="cpu")
    fingerprint: str = field(default="")
    source_tap: str = field(default="semantic")

    def __post_init__(self) -> None:
        if self.source_tap not in _VALID_SOURCE_TAPS:
            raise ValueError(f"source_tap must be one of {_VALID_SOURCE_TAPS}")
        if not self.fingerprint:
            object.__setattr__(
                self,
                "fingerprint",
                _fingerprint_tensors(
                    self.global_k,
                    self.global_v,
                    self.seq_len,
                    self.memory_dim,
                    self.dtype,
                    self.device,
                    self.source_tap,
                ),
            )

    def to_dict(self) -> dict:
        return {
            "global_k": self.global_k.detach().cpu().tolist(),
            "global_v": self.global_v.detach().cpu().tolist(),
            "seq_len": self.seq_len,
            "memory_dim": self.memory_dim,
            "dtype": self.dtype,
            "device": self.device,
            "fingerprint": self.fingerprint,
            "source_tap": self.source_tap,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EncoderMemory":
        try:
            dtype = _DTYPE_BY_NAME[payload["dtype"]]
        except KeyError:
            raise ValueError(f"unknown dtype tag: {payload.get('dtype')!r}")
        global_k = torch.tensor(payload["global_k"], dtype=dtype)
        global_v = torch.tensor(payload["global_v"], dtype=dtype)
        candidate = cls(
            global_k=global_k,
            global_v=global_v,
            seq_len=int(payload["seq_len"]),
            memory_dim=int(payload["memory_dim"]),
            dtype=str(payload["dtype"]),
            device=str(payload["device"]),
            fingerprint="",
            source_tap=str(payload["source_tap"]),
        )
        if candidate.fingerprint != payload["fingerprint"]:
            raise ValueError("EncoderMemory fingerprint mismatch: payload was tampered")
        candidate.fingerprint = str(payload["fingerprint"])
        return candidate


@dataclass
class LayerMemoryView:
    """Lightweight per-layer view sharing the parent memory storage."""

    layer_index: int
    k: torch.Tensor
    v: torch.Tensor


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight.to(dtype=x.dtype)


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


class EncoderMemoryBridge(nn.Module):
    """RMSNorm -> proj -> SwiGLU residual adapter -> shared global K/V."""

    def __init__(
        self,
        encoder_dim: int = ENCODER_HIDDEN_SIZE,
        memory_dim: int = DECODER_HIDDEN_SIZE,
        detail_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.encoder_dim = _require_positive_int("encoder_dim", encoder_dim)
        self.memory_dim = _require_positive_int("memory_dim", memory_dim)
        self.detail_dim = _require_positive_int(
            "detail_dim", detail_dim if detail_dim is not None else encoder_dim
        )
        self.dtype = dtype
        self.device = device

        self.norm = _RMSNorm(self.encoder_dim, dtype=dtype, device=device)
        self.proj = nn.Linear(
            self.encoder_dim, self.memory_dim, bias=False, dtype=dtype, device=device
        )
        adapter_hidden = max(8, self.memory_dim // 4)
        self.adapt_gate_proj = nn.Linear(
            self.memory_dim, adapter_hidden, bias=False, dtype=dtype, device=device
        )
        self.adapt_up_proj = nn.Linear(
            self.memory_dim, adapter_hidden, bias=False, dtype=dtype, device=device
        )
        self.adapt_down_proj = nn.Linear(
            adapter_hidden, self.memory_dim, bias=False, dtype=dtype, device=device
        )
        with torch.no_grad():
            self.adapt_down_proj.weight.zero_()
        self.output_scale = nn.Parameter(torch.zeros((), dtype=dtype, device=device))
        self.detail_norm = _RMSNorm(self.detail_dim, dtype=dtype, device=device)
        self.detail_proj = nn.Linear(
            self.detail_dim, self.memory_dim, bias=False, dtype=dtype, device=device
        )
        self.k_proj = nn.Linear(
            self.memory_dim, self.memory_dim, bias=False, dtype=dtype, device=device
        )
        self.v_proj = nn.Linear(
            self.memory_dim, self.memory_dim, bias=False, dtype=dtype, device=device
        )
        # Exact 0.0: gated contribution is exactly zero until training opens it.
        self.output_gate = nn.Parameter(torch.zeros((), dtype=dtype, device=device))

    # -- taps -----------------------------------------------------------------
    def _checked_input(self, states: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
        if not isinstance(states, torch.Tensor) or states.dim() != 3:
            raise ValueError(f"{name} must be a [batch, seq, dim] tensor")
        if states.shape[-1] != expected_dim:
            raise ValueError(f"{name} last dim must be {expected_dim}, got {states.shape[-1]}")
        casted = states.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(casted).all()):
            raise ValueError(f"{name} contains non-finite values")
        return casted

    def build_semantic(self, encoder_final_states: torch.Tensor) -> torch.Tensor:
        x = self._checked_input(encoder_final_states, self.encoder_dim, "encoder_final_states")
        return self._adapted(self.proj(self.norm(x)))

    def build_detail(self, encoder_intermediate_states: torch.Tensor) -> torch.Tensor:
        x = self._checked_input(
            encoder_intermediate_states, self.detail_dim, "encoder_intermediate_states"
        )
        return self.detail_proj(self.detail_norm(x))

    def _adapted(self, x: torch.Tensor) -> torch.Tensor:
        gated = nn.functional.silu(self.adapt_gate_proj(x)) * self.adapt_up_proj(x)
        return x + self.output_scale * self.adapt_down_proj(gated)

    # -- fusion / memory --------------------------------------------------------
    def build_memory(
        self, semantic: torch.Tensor, detail: torch.Tensor | None = None
    ) -> EncoderMemory:
        sem = self._checked_input(semantic, self.memory_dim, "semantic")
        if detail is None:
            fused, tap = sem, "semantic"
        else:
            det = self._checked_input(detail, self.memory_dim, "detail")
            fused, tap = sem + det.mean(dim=1, keepdim=True), "detail"
        return self._project_memory(fused, tap, gate_value=float(self.output_gate.detach()))

    def forward(
        self,
        semantic_states: torch.Tensor,
        detail_states: torch.Tensor | None = None,
        gate_enabled: bool = True,
    ) -> EncoderMemory:
        sem = self.build_semantic(semantic_states)
        det = self.build_detail(detail_states) if detail_states is not None else None
        fused = sem if det is None else sem + det.mean(dim=1, keepdim=True)
        tap = "detail" if det is not None else "semantic"
        gate = float(self.output_gate.detach()) if gate_enabled else 0.0
        return self._project_memory(fused, tap, gate_value=gate)

    def _project_memory(
        self, fused: torch.Tensor, source_tap: str, gate_value: float
    ) -> EncoderMemory:
        raw_k = self.k_proj(fused)
        raw_v = self.v_proj(fused)
        scale = torch.tensor(gate_value, dtype=raw_k.dtype, device=raw_k.device)
        global_k = (raw_k * scale).contiguous()
        global_v = (raw_v * scale).contiguous()
        seq_len = global_k.shape[1]
        return EncoderMemory(
            global_k=global_k,
            global_v=global_v,
            seq_len=seq_len,
            memory_dim=self.memory_dim,
            dtype=str(self.dtype),
            device=str(self.device),
            fingerprint="",
            source_tap=source_tap,
        )

    # -- reuse -------------------------------------------------------------------
    def reuse_for_layers(self, memory: EncoderMemory, n_layers: int) -> list[LayerMemoryView]:
        _require_positive_int("n_layers", n_layers)
        return [
            LayerMemoryView(
                layer_index=i,
                k=memory.global_k.view(memory.global_k.shape),
                v=memory.global_v.view(memory.global_v.shape),
            )
            for i in range(n_layers)
        ]

    def blank_memory_like(self, memory: EncoderMemory) -> EncoderMemory:
        return EncoderMemory(
            global_k=torch.zeros_like(memory.global_k),
            global_v=torch.zeros_like(memory.global_v),
            seq_len=memory.seq_len,
            memory_dim=memory.memory_dim,
            dtype=memory.dtype,
            device=memory.device,
            fingerprint="",
            source_tap=memory.source_tap,
        )

    # -- rejected paths ------------------------------------------------------------
    def splice_into_decoder_states(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError(
            "direct hidden-state splicing is rejected: the bridge only emits "
            "K/V memory; use EncoderMemory via the decoder-side hook"
        )

    # -- gating / training state -----------------------------------------------------
    def freeze(self) -> "EncoderMemoryBridge":
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> "EncoderMemoryBridge":
        for param in self.parameters():
            param.requires_grad_(True)
        return self

    def trainable_parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # -- serialization -----------------------------------------------------------------
    def state_dict_snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def metadata(self) -> dict:
        digest = hashlib.sha256()
        digest.update(f"{self.encoder_dim}|{self.memory_dim}|{self.detail_dim}|".encode())
        digest.update(f"{str(self.dtype)}|{str(self.device)}".encode())
        for key in sorted(self.state_dict()):
            tensor = self.state_dict()[key].detach().to("cpu", torch.float32).contiguous()
            digest.update(tensor.numpy().tobytes())
        return {
            "encoder_dim": self.encoder_dim,
            "memory_dim": self.memory_dim,
            "detail_dim": self.detail_dim,
            "taps": ["semantic", "detail"],
            "dtype": str(self.dtype),
            "device": str(self.device),
            "fingerprint": digest.hexdigest(),
        }

    def load_snapshot_strict(self, state: dict[str, torch.Tensor], metadata: dict) -> None:
        expected = {
            "encoder_dim": self.encoder_dim,
            "memory_dim": self.memory_dim,
            "detail_dim": self.detail_dim,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"snapshot metadata mismatch for {key!r}: got {metadata.get(key)!r}, "
                    f"expected {want!r}"
                )
        self.load_state_dict(state, strict=True)

    # -- validation interfaces --------------------------------------------------------------
    def dense_reference(self, memory: EncoderMemory) -> tuple[torch.Tensor, torch.Tensor]:
        return memory.global_k, memory.global_v

    def sparse_memory_stub(
        self, memory: EncoderMemory, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_positive_int("top_k", top_k)
        if top_k > memory.seq_len:
            raise ValueError(f"top_k={top_k} exceeds seq_len={memory.seq_len}")
        magnitudes = memory.global_k.norm(dim=-1)
        indices = magnitudes.topk(min(top_k, magnitudes.shape[-1]), dim=-1).indices
        batch = torch.arange(memory.global_k.shape[0], device=memory.global_k.device)
        gathered_k = memory.global_k[batch[:, None], indices]
        gathered_v = memory.global_v[batch[:, None], indices]
        return indices, gathered_k, gathered_v
