"""Real OpenReasoning-Nemotron loader and attention-transplant wrapper.

The tiny ``KestrelForCausalLM`` is intentionally test-only.  This module
loads the actual Qwen2/Nemotron checkpoint through Transformers, keeps the
original embeddings, norms, FFNs, and LM head frozen (4-bit on local CUDA),
and replaces only self-attention with the V4-Flash reference transplant.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention.cache import KestrelCache
from .attention.mhc import ManifoldHyperConnection
from .attention.module import V4FlashAttention
from .configuration import KestrelConfig
from .multimodal_model import KestrelOutput
from .transplant.svd_init import initialize_attention_from_dense


def _materialize_linear_weight(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Return a dense temporary copy, including a bitsandbytes 4-bit layer."""
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        from bitsandbytes.functional import dequantize_4bit

        return dequantize_4bit(weight.data, quant_state=quant_state).to(dtype)
    return weight.detach().to(dtype)


class RealDecoderLayer(nn.Module):
    def __init__(
        self,
        input_norm: nn.Module,
        post_norm: nn.Module,
        mlp: nn.Module,
        attention: V4FlashAttention,
        config: KestrelConfig,
    ) -> None:
        super().__init__()
        self.input_norm = input_norm
        self.attention = attention
        self.attn_mhc = ManifoldHyperConnection(
            config.mhc_streams, config.mhc_sinkhorn_iters, config.mhc_enabled
        ).to(device=next(attention.parameters()).device, dtype=next(attention.parameters()).dtype)
        self.post_norm = post_norm
        self.mlp = mlp
        self.mlp_mhc = ManifoldHyperConnection(
            config.mhc_streams, config.mhc_sinkhorn_iters, config.mhc_enabled
        ).to(device=next(attention.parameters()).device, dtype=next(attention.parameters()).dtype)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, cache: KestrelCache | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn, branch = self.attention(self.input_norm(x), position_ids, cache)
        x = self.attn_mhc(x, attn)
        update = self.mlp(self.post_norm(x))
        x = self.mlp_mhc(x, update)
        return x, branch


class RealNemotronKestrelForCausalLM(nn.Module):
    """Nemotron body plus the reference V4-style attention transplant."""

    def __init__(
        self,
        config: KestrelConfig,
        embed_tokens: nn.Module,
        layers: list[RealDecoderLayer],
        norm: nn.Module,
        lm_head: nn.Module,
        base_model_id: str,
        base_revision: str | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head
        self.base_model_id = base_model_id
        self.base_revision = base_revision

    def freeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for layer in self.layers:
            for parameter in layer.attention.parameters():
                parameter.requires_grad_(True)
            if self.config.mhc_enabled:
                for parameter in layer.attn_mhc.parameters():
                    parameter.requires_grad_(True)
                for parameter in layer.mlp_mhc.parameters():
                    parameter.requires_grad_(True)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        missing = []
        for name, parameter in own.items():
            if parameter.requires_grad:
                if name not in state:
                    missing.append(name)
                else:
                    parameter.data.copy_(state[name].to(parameter.device, parameter.dtype))
        if missing:
            raise KeyError(f"checkpoint is missing trainable parameters, first={missing[:3]}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KestrelCache | None = None,
        return_dict: bool = True,
    ) -> KestrelOutput | tuple[torch.Tensor, torch.Tensor | None]:
        del attention_mask
        x = self.embed_tokens(input_ids)
        if position_ids is None:
            start = past_key_values.length(0) if past_key_values is not None else 0
            position_ids = torch.arange(start, start + x.shape[1], device=x.device).view(1, -1)
            position_ids = position_ids.expand(x.shape[0], -1)
        branch = None
        for layer in self.layers:
            x, branch = layer(x, position_ids, past_key_values)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        output = KestrelOutput(logits=logits, loss=loss, past_key_values=past_key_values)
        return output if return_dict else (logits, loss)

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )


@dataclass(frozen=True)
class NemotronLoadInfo:
    model_id: str
    revision: str | None
    quantized: bool
    dtype: str
    parameter_count: int
    trainable_parameter_count: int
    initialization_errors: tuple[dict[str, float], ...]


def load_real_nemotron_transplant(
    config: KestrelConfig,
    model_id: str = "nvidia/OpenReasoning-Nemotron-7B",
    revision: str | None = None,
    device: str | torch.device = "cuda",
    load_in_4bit: bool = True,
    compute_dtype: torch.dtype = torch.float16,
) -> tuple[RealNemotronKestrelForCausalLM, NemotronLoadInfo]:
    """Load real Nemotron weights and construct one transplant candidate."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    target = torch.device(device)
    if load_in_4bit and target.type != "cuda":
        raise ValueError("4-bit bitsandbytes loading is only supported on CUDA")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": target.index if target.index is not None else 0}
    else:
        kwargs["device_map"] = {"": target.index if target.index is not None else 0}
    base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    hf_config = base.config
    if int(hf_config.hidden_size) != 3584 or int(hf_config.num_hidden_layers) != 28:
        raise ValueError("the requested base is not the expected Nemotron-7B geometry")
    old_kv_heads = int(hf_config.num_key_value_heads)
    layers: list[RealDecoderLayer] = []
    errors: list[dict[str, float]] = []
    for index, old_layer in enumerate(base.model.layers):
        attention = V4FlashAttention(config, index).to(device=target, dtype=compute_dtype)
        old_attention = old_layer.self_attn
        dense_weights = tuple(
            _materialize_linear_weight(getattr(old_attention, name), compute_dtype)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        errors.append(
            initialize_attention_from_dense(
                attention, dense_weights[0], dense_weights[1], dense_weights[2], dense_weights[3], old_kv_heads
            )
        )
        layers.append(
            RealDecoderLayer(
                old_layer.input_layernorm,
                old_layer.post_attention_layernorm,
                old_layer.mlp,
                attention,
                config,
            )
        )
        del old_attention, dense_weights, attention
        if target.type == "cuda":
            torch.cuda.empty_cache()
    model = RealNemotronKestrelForCausalLM(
        config,
        base.model.embed_tokens,
        layers,
        base.model.norm,
        base.lm_head,
        model_id,
        revision,
    )
    del base
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    model.freeze_backbone()
    info = NemotronLoadInfo(
        model_id=model_id,
        revision=revision,
        quantized=load_in_4bit,
        dtype=str(compute_dtype),
        parameter_count=model.parameter_count(),
        trainable_parameter_count=model.parameter_count(trainable_only=True),
        initialization_errors=tuple(errors),
    )
    return model, info
