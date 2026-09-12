"""Heterogeneous CED runtime doubles (issue #3).

Tiny CPU-friendly stand-ins for the Qwen3.5-2B-Base encoder and Qwen3.5-9B
decoder plus the gated external-memory hook. No HF downloads, no network:
real-model loading is represented by the strict state-dict path together
with :class:`CEDSourceConfig` revision pinning.

Contract constants live in :mod:`model.ced_config` and are imported here,
never redefined.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.ced_config import (
    CEDSourceConfig,
    ZERO_GATE_LOGIT_TOL,
    assert_no_hidden_size_splice,
)

__all__ = [
    "PromptEncoding",
    "TinyCausalEncoder",
    "TinyAutoregressiveDecoder",
    "ExternalMemoryHook",
    "CEDRuntime",
    "ZERO_GATE_LOGIT_TOL",
]


def _sha256_tensor(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _check_strict_keys(
    expected: List[str], provided: Mapping[str, Tensor], owner: str, strict: bool
) -> tuple[List[str], List[str]]:
    expected_set = set(expected)
    provided_set = set(provided.keys())
    missing = sorted(expected_set - provided_set)
    unexpected = sorted(provided_set - expected_set)
    if strict and missing:
        raise ValueError(f"{owner}: missing keys in strict load: {missing}")
    if unexpected:
        raise ValueError(f"{owner}: unexpected keys in load: {unexpected}")
    inherited = sorted(expected_set & provided_set)
    new = sorted(expected_set - provided_set)
    return inherited, new


@dataclass
class PromptEncoding:
    """Output of the encoder phase, consumed by the memory-conditioned decode."""

    input_ids: Tensor
    memory: Tensor


class TinyCausalEncoder(nn.Module):
    """Tiny bidirectional-style encoder double with a strict load path."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        vocab_size: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(hidden_dim, hidden_dim),
                        "norm": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.inherited_keys: List[str] = []
        self.new_keys: List[str] = list(self.state_dict().keys())
        self._num_heads = num_heads

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer["norm"](hidden + layer["lin"](hidden))
        return self.final_norm(hidden)

    def load_source_state(self, checkpoint: Mapping[str, Tensor], strict: bool = True) -> None:
        """Deterministic from-pretrained-like init; never silently re-inits."""
        inherited, new = _check_strict_keys(
            list(self.state_dict().keys()), checkpoint, "TinyCausalEncoder", strict
        )
        self.load_state_dict(dict(checkpoint), strict=strict)
        self.inherited_keys = inherited
        self.new_keys = new

    def parameter_checksums(self) -> Dict[str, str]:
        return {name: _sha256_tensor(p) for name, p in self.named_parameters()}


class TinyAutoregressiveDecoder(nn.Module):
    """Tiny autoregressive decoder double with a strict load path."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        vocab_size: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(hidden_dim, hidden_dim),
                        "norm": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.inherited_keys: List[str] = []
        self.new_keys: List[str] = list(self.state_dict().keys())
        self._num_heads = num_heads

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer["norm"](hidden + layer["lin"](hidden))
        return self.final_norm(hidden)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.forward_hidden(input_ids))

    def load_source_state(self, checkpoint: Mapping[str, Tensor], strict: bool = True) -> None:
        """Deterministic from-pretrained-like init; never silently re-inits."""
        inherited, new = _check_strict_keys(
            list(self.state_dict().keys()), checkpoint, "TinyAutoregressiveDecoder", strict
        )
        self.load_state_dict(dict(checkpoint), strict=strict)
        self.inherited_keys = inherited
        self.new_keys = new

    def parameter_checksums(self) -> Dict[str, str]:
        return {name: _sha256_tensor(p) for name, p in self.named_parameters()}


class ExternalMemoryHook(nn.Module):
    """Gated additive memory contribution to decoder hidden states.

    Output = hidden + gate * proj(pool(memory)). The gate is a scalar
    parameter initialised to EXACTLY 0.0 so the hook is a no-op at init.
    """

    def __init__(self, decoder_dim: int = 32, encoder_dim: int = 32, enabled: bool = True) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.enabled = enabled
        self.gate = nn.Parameter(torch.zeros(1))
        self.memory_proj = nn.Linear(encoder_dim, decoder_dim)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def forward(
        self,
        decoder_hidden: Tensor,
        memory: Tensor,
        gate_scale: Optional[float | Tensor] = None,
        enabled: Optional[bool] = None,
    ) -> Tensor:
        active = self.enabled if enabled is None else bool(enabled)
        if not active:
            return decoder_hidden
        scale = self.gate if gate_scale is None else gate_scale
        if isinstance(scale, Tensor):
            if scale.numel() == 1 and float(scale.detach().cpu()) == 0.0:
                return decoder_hidden
        elif scale == 0.0:
            return decoder_hidden
        pooled = memory.mean(dim=1, keepdim=True) if memory.dim() == 3 else memory
        contribution = scale * self.memory_proj(pooled)
        return decoder_hidden + contribution


class CEDRuntime(nn.Module):
    """Wraps encoder + decoder + memory hook with explicit phased inference."""

    def __init__(
        self,
        encoder: TinyCausalEncoder,
        decoder: TinyAutoregressiveDecoder,
        memory_hook: ExternalMemoryHook,
        encoder_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        source_config: Optional[CEDSourceConfig] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.memory_hook = memory_hook
        resolved_enc = encoder.hidden_dim if encoder_dim is None else encoder_dim
        resolved_dec = decoder.hidden_dim if decoder_dim is None else decoder_dim
        if resolved_enc != encoder.hidden_dim:
            raise ValueError(
                f"encoder_dim mismatch: runtime={resolved_enc} "
                f"module={encoder.hidden_dim}; never truncate/pad."
            )
        if resolved_dec != decoder.hidden_dim:
            raise ValueError(
                f"decoder_dim mismatch: runtime={resolved_dec} "
                f"module={decoder.hidden_dim}; never truncate/pad."
            )
        if resolved_enc != memory_hook.encoder_dim or resolved_dec != memory_hook.decoder_dim:
            raise ValueError(
                "memory hook dims disagree with runtime dims "
                f"(hook={memory_hook.encoder_dim}/{memory_hook.decoder_dim}, "
                f"runtime={resolved_enc}/{resolved_dec})."
            )
        self.encoder_dim = resolved_enc
        self.decoder_dim = resolved_dec
        self.source_config = source_config or CEDSourceConfig()
        self.encode_calls = 0

    @property
    def hook(self) -> ExternalMemoryHook:
        return self.memory_hook

    # -- phased inference -------------------------------------------------
    def encode_prompt(self, input_ids: Tensor) -> PromptEncoding:
        self.encode_calls += 1
        return PromptEncoding(input_ids=input_ids, memory=self.encoder(input_ids))

    def decode_with_memory(self, input_ids: Tensor, encoding: PromptEncoding | Tensor) -> Tensor:
        hidden = self.decoder.forward_hidden(input_ids)
        memory = encoding.memory if isinstance(encoding, PromptEncoding) else encoding
        hooked = self.memory_hook(hidden, memory)
        return self.decoder.lm_head(hooked)

    def decode_without_memory(self, input_ids: Tensor) -> Tensor:
        return self.decoder(input_ids)

    # -- trainability ------------------------------------------------------
    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_decoder(self) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = True

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_hook.set_enabled(enabled)

    def trainable_parameter_counts(self) -> Dict[str, int]:
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        enc = _count(self.encoder)
        dec = _count(self.decoder)
        hook = _count(self.memory_hook)
        return {"encoder": enc, "decoder": dec, "hook": hook, "total": enc + dec + hook}

    # -- contracts ----------------------------------------------------------
    def assert_hidden_contract(self) -> None:
        assert_no_hidden_size_splice(self.encoder_dim, self.decoder_dim)

    def validate_sources(self, strict_revisions: bool = True) -> None:
        self.source_config.validate_architecture()
        if strict_revisions:
            self.source_config.require_pinned_revisions()

    def parameter_checksums(self) -> Dict[str, str]:
        sums: Dict[str, str] = {}
        for prefix, module in (
            ("encoder.", self.encoder),
            ("decoder.", self.decoder),
            ("hook.", self.memory_hook),
        ):
            for name, param in module.named_parameters():
                sums[f"{prefix}{name}"] = _sha256_tensor(param)
        combined = hashlib.sha256(
            "\n".join(f"{k}:{sums[k]}" for k in sorted(sums)).encode("utf-8")
        ).hexdigest()
        sums["_combined"] = combined
        return sums
