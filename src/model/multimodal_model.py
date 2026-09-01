"""Compact HF-friendly model shell used for smoke tests and later transplant."""

import math
import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention.cache import KestrelCache
from .attention.mhc import ManifoldHyperConnection
from .attention.module import V4FlashAttention
from .configuration import KestrelConfig
from .vision.internvit import InternViTEncoder, dynamic_tiles
from .vision.projector import AdaptiveVisionProjector


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: KestrelConfig, index: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = V4FlashAttention(config, index)
        self.attn_mhc = ManifoldHyperConnection(config.mhc_streams, config.mhc_sinkhorn_iters, config.mhc_enabled)
        self.post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.mlp_mhc = ManifoldHyperConnection(config.mhc_streams, config.mhc_sinkhorn_iters, config.mhc_enabled)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cache: KestrelCache | None) -> tuple[torch.Tensor, torch.Tensor]:
        attn, branch = self.attention(self.input_norm(x), position_ids, cache)
        x = self.attn_mhc(x, attn)
        update = self.mlp(self.post_norm(x))
        x = self.mlp_mhc(x, update)
        return x, branch


@dataclass
class KestrelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KestrelCache | None = None
    hidden_states: torch.Tensor | None = None


class KestrelForCausalLM(nn.Module):
    def __init__(self, config: KestrelConfig | None = None, vision: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or KestrelConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(self.config, i) for i in range(self.config.num_hidden_layers))
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.vision_encoder = vision or (InternViTEncoder(hidden_size=self.config.vision_hidden_size) if self.config.use_vision else None)
        self.vision_projector = AdaptiveVisionProjector(self.config.vision_hidden_size, self.config.hidden_size, self.config.vision_token_budget) if self.config.use_vision else None
        self._vision_encoded_cache: dict[str, torch.Tensor] = {}
        self.last_vision_telemetry: dict[str, object] = {}

    def _default_vision_budget(self, tile_count: int, kind: str) -> int:
        named = {
            "ordinary": self.config.vision_budget_ordinary,
            "document": self.config.vision_budget_document,
            "ide": self.config.vision_budget_ide,
            "high_resolution": self.config.vision_budget_high_resolution,
        }
        if kind in named:
            return named[kind]
        if tile_count >= 8:
            return self.config.vision_budget_high_resolution
        if tile_count >= 4:
            return self.config.vision_budget_ide
        if tile_count >= 2:
            return self.config.vision_budget_document
        return self.config.vision_budget_ordinary

    def visual_tokens(
        self,
        pixel_values: torch.Tensor,
        budget: int | None = None,
        context_length: int = 0,
        kind: str = "auto",
    ) -> torch.Tensor:
        if self.vision_encoder is None or self.vision_projector is None:
            raise RuntimeError("vision is disabled")
        if pixel_values.ndim == 3:
            pixel_values = dynamic_tiles(pixel_values)
        if pixel_values.ndim == 4 and pixel_values.shape[0] > 1:
            pixels = pixel_values
        else:
            pixels = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        language_device = self.embed_tokens.weight.device
        offload = bool(
            context_length > self.config.vision_offload_threshold
            and self.config.vision_freeze_long_context
            and not any(parameter.requires_grad for parameter in self.vision_encoder.parameters())
        )
        cacheable = self.config.vision_cache_encoded and not any(
            parameter.requires_grad for parameter in self.vision_encoder.parameters()
        )
        cache_key = hashlib.sha256(
            pixels.detach().to(device="cpu").contiguous().numpy().tobytes()
        ).hexdigest()
        cache_hit = False
        if cacheable and cache_key in self._vision_encoded_cache:
            encoded = self._vision_encoded_cache[cache_key].to(language_device, non_blocking=True)
            cache_hit = True
            self.last_vision_telemetry = {
                "vision_device": "cached_cpu",
                "language_device": str(language_device),
                "offloaded": offload,
                "encoded_once": False,
                "cache_hit": True,
                "tiles": int(pixels.shape[0]),
                "tokens": int(encoded.shape[-2]),
            }
        elif hasattr(self.vision_encoder, "encode_with_policy"):
            encoded = self.vision_encoder.encode_with_policy(pixels, language_device, offload_to_cpu=offload)  # type: ignore[attr-defined]
            self.last_vision_telemetry = dict(getattr(self.vision_encoder, "last_telemetry", {}))
        else:
            encoded = self.vision_encoder(pixels.to(language_device))
            self.last_vision_telemetry = {
                "vision_device": str(language_device),
                "language_device": str(language_device),
                "offloaded": False,
                "encoded_once": True,
                "tiles": int(pixels.shape[0]),
                "tokens": int(encoded.shape[-2]),
            }
        if cacheable and not cache_hit:
            self._vision_encoded_cache[cache_key] = encoded.detach().to(device="cpu")
        self.last_vision_telemetry["cache_hit"] = cache_hit
        self.last_vision_telemetry["budget"] = budget or self._default_vision_budget(int(pixels.shape[0]), kind)
        encoded = encoded.reshape(1, -1, encoded.shape[-1])
        return self.vision_projector(encoded, self.last_vision_telemetry["budget"])  # type: ignore[arg-type]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KestrelCache | None = None,
        return_dict: bool = True,
        logits_to_keep: int | None = None,
    ) -> KestrelOutput | tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens(input_ids)
        prefix = 0
        if pixel_values is not None:
            if past_key_values is not None:
                raise ValueError("pixel_values may only be supplied during prefill")
            visual = self.visual_tokens(pixel_values, context_length=int(input_ids.shape[1]))
            x = torch.cat((visual, x), dim=1)
            prefix = visual.shape[1]
            if labels is not None:
                labels = F.pad(labels, (prefix, 0), value=-100)
        if position_ids is None:
            start = past_key_values.length(0) if past_key_values is not None else 0
            position_ids = torch.arange(start, start + x.shape[1], device=x.device).view(1, -1).expand(x.shape[0], -1)
        branch = None
        for layer in self.layers:
            x, branch = layer(x, position_ids, past_key_values)
        normalized = self.norm(x)
        if logits_to_keep is not None:
            if logits_to_keep < 1:
                raise ValueError("logits_to_keep must be positive")
            normalized = normalized[:, -logits_to_keep:]
        logits = self.lm_head(normalized)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        output = KestrelOutput(logits=logits, loss=loss, past_key_values=past_key_values)
        return output if return_dict else (logits, loss)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, pixel_values: torch.Tensor | None = None, eos_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        cache = KestrelCache()
        output = self(input_ids, pixel_values=pixel_values, past_key_values=cache)
        tokens = input_ids
        next_token = output.logits[:, -1:].argmax(dim=-1)
        for _ in range(max_new_tokens):
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
            output = self(next_token, past_key_values=cache)
            next_token = output.logits[:, -1:].argmax(dim=-1)
        return tokens

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
