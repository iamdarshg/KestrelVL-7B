"""Long-context validation helpers and cache-budget estimation.

The estimator is deliberately analytical: it reports the state retained by
the reference cache without allocating a million-token tensor.  It is useful
for selecting GPU versus CPU compressed-cache placement before a real
inference run.
"""

from __future__ import annotations

import math
from typing import Any


def _dtype_bytes(name: str) -> int:
    values = {
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported cache dtype: {name}") from exc


def estimate_cache_memory(
    config: Any,
    context_tokens: int,
    batch_size: int = 1,
    local_dtype_bytes: int = 2,
    compressed_dtype_bytes: int | None = None,
    position_dtype_bytes: int = 8,
) -> dict[str, Any]:
    """Estimate retained cache bytes for a Kestrel configuration.

    The result is an upper bound for normal decode-style append behavior:
    compressed/index chunks are counted at ``ceil(T / ratio)`` and integer
    index modes include one FP16 scale per stored compressed item.  The
    estimator does not include model weights, activations, allocator reserve,
    or temporary GEMM buffers, so a GPU deployment must leave headroom.
    """
    if context_tokens < 1 or batch_size < 1:
        raise ValueError("context_tokens and batch_size must be positive")
    if compressed_dtype_bytes is None:
        compressed_dtype_bytes = _dtype_bytes(str(config.compressed_kv_dtype))
    index_bytes = _dtype_bytes(str(config.index_dtype))
    layers = list(config.layer_schedule)
    local_tokens = min(int(config.sliding_window), context_tokens)
    local_per_layer = batch_size * (
        2 * int(config.num_key_value_heads) * local_tokens * int(config.head_dim) * local_dtype_bytes
        + local_tokens * position_dtype_bytes
    )
    totals = {
        "local_kv": 0,
        "pending_compression": 0,
        "compressed_state": 0,
        "index_state": 0,
    }
    per_layer: list[dict[str, Any]] = []
    for layer_index, mode in enumerate(layers):
        compressed = 0
        pending = 0
        index = 0
        ratio = None
        if mode in {"csa", "hca"}:
            ratio = int(
                config.csa_compression_ratio
                if mode == "csa"
                else config.hca_compression_ratio
            )
            compressed = math.ceil(context_tokens / ratio)
            pending_tokens = min(context_tokens, max(0, ratio - 1))
            pending = batch_size * 2 * int(config.num_key_value_heads) * pending_tokens * int(
                config.head_dim
            ) * local_dtype_bytes
            pending += batch_size * pending_tokens * position_dtype_bytes
            compressed_bytes = batch_size * (
                2
                * int(config.num_key_value_heads)
                * compressed
                * int(config.head_dim)
                * compressed_dtype_bytes
                + compressed * position_dtype_bytes
            )
            index = batch_size * int(config.num_attention_heads) * compressed * int(
                config.index_head_dim
            ) * index_bytes
            if str(config.index_dtype).startswith("int"):
                # Decode can create one scale tensor per emitted compressed
                # chunk.  Full-prefill implementations often use fewer
                # scales, so this is intentionally the safe upper bound.
                index += batch_size * int(config.num_attention_heads) * compressed * 2
        else:
            compressed_bytes = 0
        totals["local_kv"] += local_per_layer
        totals["pending_compression"] += pending
        totals["compressed_state"] += compressed_bytes
        totals["index_state"] += index
        per_layer.append(
            {
                "layer": layer_index,
                "mode": mode,
                "ratio": ratio,
                "local_tokens": local_tokens,
                "compressed_tokens": compressed,
                "bytes": {
                    "local_kv": local_per_layer,
                    "pending_compression": pending,
                    "compressed_state": compressed_bytes,
                    "index_state": index,
                    "total": local_per_layer + pending + compressed_bytes + index,
                },
            }
        )
    totals["total"] = sum(totals.values())
    return {
        "context_tokens": context_tokens,
        "batch_size": batch_size,
        "index_dtype": str(config.index_dtype),
        "compressed_kv_dtype": str(config.compressed_kv_dtype),
        "bytes": totals,
        "gib": {key: value / 2**30 for key, value in totals.items()},
        "per_layer": per_layer,
        "includes_model_weights": False,
        "includes_peak_activations": False,
        "scale_bound": "one_fp16_scale_per_compressed_item_for_integer_index_modes",
    }


def niah_score(prediction: str, needle: str) -> float:
    return float(needle in prediction)
