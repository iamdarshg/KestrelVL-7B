"""Real OpenReasoning-Nemotron loader and attention-transplant wrapper.

The tiny ``KestrelForCausalLM`` is intentionally test-only.  This module
loads the actual Qwen2/Nemotron checkpoint through Transformers, keeps the
original embeddings, norms, FFNs, and LM head frozen (4-bit on local CUDA),
and replaces only self-attention with the V4-Flash reference transplant.
"""

from __future__ import annotations

import gc
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention.cache import KestrelCache
from .attention.mhc import ManifoldHyperConnection
from .attention.module import V4FlashAttention
from .configuration import KestrelConfig
from .multimodal_model import KestrelOutput
from .transplant.svd_init import initialize_attention_from_dense


def _materialize_linear_weight(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Return a dense temporary copy, including a bitsandbytes 4-bit layer."""
    weight = module.weight
    # bitsandbytes has exposed quant_state on Params4bit and, in some
    # Transformers/bitsandbytes combinations, on the owning module instead.
    quant_state = getattr(weight, "quant_state", None) or getattr(module, "quant_state", None)
    if quant_state is not None:
        from bitsandbytes.functional import dequantize_4bit

        return dequantize_4bit(weight.data, quant_state=quant_state).to(dtype)
    return weight.detach().to(dtype)


class _StreamingShardReader:
    """Read one safetensors tensor at a time without a 15 GB CPU peak.

    Transformers' normal sharded loader is convenient on large hosts, but it
    briefly creates enough virtual/CPU pressure to fail on the 16 GB Windows
    developer machine.  This reader is deliberately boring: the index is
    loaded once, one shard is opened for one tensor, and the tensor is released
    as soon as its destination module owns it.  It is also useful on small
    cloud workers because it bounds host memory independently of shard size.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, dict[str, dict[str, Any]]] = {}
        for shard in sorted(set(self.weight_map.values())):
            with (root / shard).open("rb") as handle:
                header_size_raw = handle.read(8)
                header_size = struct.unpack("<Q", header_size_raw)[0]
                header = json.loads(handle.read(header_size).decode("utf-8"))
            self.headers[shard] = {
                name: value for name, value in header.items() if name != "__metadata__"
            }

    def cpu(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        shard = self.weight_map[name]
        entry = self.headers[shard][name]
        start, end = (int(value) for value in entry["data_offsets"])
        # Safetensors offsets are relative to the start of the payload, after
        # the 8-byte header length and JSON header.  Reading the exact range
        # avoids Windows mmap/pagefile pressure from mapping a 10 GB shard.
        with (self.root / shard).open("rb") as handle:
            header_size_raw = handle.read(8)
            header_size = struct.unpack("<Q", header_size_raw)[0]
            handle.seek(8 + header_size + start)
            raw = bytearray(handle.read(end - start))
        storage_dtype = {
            "BF16": torch.uint16,
            "F16": torch.float16,
            "F32": torch.float32,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }[entry["dtype"]]
        tensor = torch.frombuffer(raw, dtype=storage_dtype)
        if entry["dtype"] == "BF16":
            tensor = tensor.view(torch.bfloat16)
        tensor = tensor.reshape(tuple(int(value) for value in entry["shape"]))
        return tensor.to(dtype=dtype) if dtype is not None and tensor.dtype != dtype else tensor


def _replace_with_4bit_linear(
    old: nn.Module,
    weight: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct a bitsandbytes linear from a single streamed CPU tensor."""
    import bitsandbytes as bnb

    if not isinstance(old, nn.Linear):
        raise TypeError(f"expected Linear leaf, got {type(old)!r}")
    linear = bnb.nn.Linear4bit(
        old.in_features,
        old.out_features,
        bias=old.bias is not None,
        compute_dtype=dtype,
        quant_type="nf4",
        compress_statistics=True,
    ).to(device)
    linear.weight = bnb.nn.Params4bit(
        weight.detach().to(device="cpu", dtype=dtype),
        requires_grad=False,
        quant_type="nf4",
        compress_statistics=True,
    )
    if old.bias is not None:
        linear.bias = nn.Parameter(old.bias.detach().to(device=device, dtype=dtype), requires_grad=False)
    # Params4bit quantizes when moved to CUDA.  Keep this second .to explicit:
    # it is required by bnb 0.49 and is a no-op after quantization.
    return linear.to(device)


def _stream_mlp(
    old_mlp: nn.Module,
    reader: _StreamingShardReader,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Materialize the three frozen Qwen MLP projections as NF4 linears."""
    prefix = f"model.layers.{layer_idx}.mlp"
    for name in ("gate_proj", "up_proj", "down_proj"):
        old = getattr(old_mlp, name)
        weight = reader.cpu(f"{prefix}.{name}.weight", dtype=dtype)
        setattr(old_mlp, name, _replace_with_4bit_linear(old, weight, device, dtype))
        del weight
    return old_mlp


def _materialize_parameter(module: nn.Module, weight: torch.Tensor, device: torch.device) -> nn.Module:
    """Attach a streamed dense parameter to a meta-initialized module."""
    module.weight = nn.Parameter(weight.to(device=device), requires_grad=False)
    return module


def _load_streaming_skeleton(
    model_id: str,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[nn.Module, _StreamingShardReader]:
    """Build a meta Qwen skeleton and attach only streamed frozen modules."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    root = Path(model_id)
    reader = _StreamingShardReader(root)
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(hf_config, torch_dtype=compute_dtype)
    return base, reader


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
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Checkpoint decoder-layer activations for the 8 GiB local profile."""
        self.gradient_checkpointing = enabled

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
            if self.training and self.gradient_checkpointing and past_key_values is None:
                x, branch = checkpoint(
                    lambda hidden, current_layer=layer: current_layer(hidden, position_ids, None),
                    x,
                    use_reentrant=False,
                )
            else:
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
    skip_svd_initialization: bool = False,
) -> tuple[RealNemotronKestrelForCausalLM, NemotronLoadInfo]:
    """Load real Nemotron weights and construct one transplant candidate."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    target = torch.device(device)
    if load_in_4bit and target.type != "cuda":
        raise ValueError("4-bit bitsandbytes loading is only supported on CUDA")
    local_root = Path(model_id)
    streaming = local_root.is_dir() and (local_root / "model.safetensors.index.json").exists()
    kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if streaming:
        # The ordinary Transformers loader has a large transient CPU/pagefile
        # peak on Windows.  The streaming path below quantizes frozen MLP and
        # LM-head leaves one tensor at a time while keeping the real weights.
        base, reader = _load_streaming_skeleton(model_id, target, compute_dtype)
    else:
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        kwargs["device_map"] = {"": target.index if target.index is not None else 0}
        base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        reader = None
    hf_config = base.config
    if int(hf_config.hidden_size) != 3584 or int(hf_config.num_hidden_layers) != 28:
        raise ValueError("the requested base is not the expected Nemotron-7B geometry")
    old_kv_heads = int(hf_config.num_key_value_heads)
    layers: list[RealDecoderLayer] = []
    errors: list[dict[str, float]] = []
    for index, old_layer in enumerate(base.model.layers):
        if reader is not None:
            print(f"streaming Nemotron layer {index + 1}/28", flush=True)
        attention = V4FlashAttention(config, index).to(device=target, dtype=compute_dtype)
        if reader is not None:
            if skip_svd_initialization:
                dense_weights = None
            else:
                dense_weights = tuple(
                    # Keep SVD workspace on the host.  A CUDA full SVD leaves
                    # a large cuSOLVER allocator footprint on 8 GiB GPUs;
                    # factors are copied into the student below.
                    reader.cpu(f"model.layers.{index}.self_attn.{name}.weight", dtype=compute_dtype)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            input_norm = _materialize_parameter(
                old_layer.input_layernorm,
                reader.cpu(f"model.layers.{index}.input_layernorm.weight", dtype=compute_dtype),
                target,
            )
            post_norm = _materialize_parameter(
                old_layer.post_attention_layernorm,
                reader.cpu(f"model.layers.{index}.post_attention_layernorm.weight", dtype=compute_dtype),
                target,
            )
            mlp = _stream_mlp(old_layer.mlp, reader, index, target, compute_dtype)
            print(f"streamed frozen MLP layer {index + 1}/28", flush=True)
        else:
            old_attention = old_layer.self_attn
            dense_weights = tuple(
                _materialize_linear_weight(getattr(old_attention, name), compute_dtype)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            input_norm, post_norm, mlp = old_layer.input_layernorm, old_layer.post_attention_layernorm, old_layer.mlp
        if dense_weights is None:
            errors.append({"cached_initialization": 1.0})
        else:
            errors.append(
                initialize_attention_from_dense(
                    attention,
                    dense_weights[0],
                    dense_weights[1],
                    dense_weights[2],
                    dense_weights[3],
                    old_kv_heads,
                )
            )
        if reader is not None:
            print(f"initialized transplant layer {index + 1}/28", flush=True)
        layers.append(
            RealDecoderLayer(
                input_norm,
                post_norm,
                mlp,
                attention,
                config,
            )
        )
        del dense_weights, attention
        if target.type == "cuda":
            torch.cuda.empty_cache()
    if reader is not None:
        embed_tokens = nn.Embedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
            _weight=reader.cpu("model.embed_tokens.weight", dtype=compute_dtype).to(target),
        )
        norm = _materialize_parameter(
            base.model.norm,
            reader.cpu("model.norm.weight", dtype=compute_dtype),
            target,
        )
        lm_head = _replace_with_4bit_linear(
            base.lm_head,
            reader.cpu("lm_head.weight", dtype=compute_dtype),
            target,
            compute_dtype,
        )
    else:
        embed_tokens, norm, lm_head = base.model.embed_tokens, base.model.norm, base.lm_head
    model = RealNemotronKestrelForCausalLM(
        config,
        embed_tokens,
        layers,
        norm,
        lm_head,
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
