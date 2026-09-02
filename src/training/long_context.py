"""Chunked long-context execution and training utilities.

These helpers make the memory/gradient trade-off explicit.  They are not a
claim that a 1M-token backward pass fits on a particular GPU; the caller gets
per-run telemetry and the exact mode used in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.attention.cache import (
    CompressedStateStore,
    IndexStateStore,
    KestrelCache,
    LayerCache,
)


@dataclass(frozen=True)
class LongContextConfig:
    mode: str = "full_recompute"
    execution_chunk_tokens: int = 8192
    detach_interval_tokens: int = 8192
    max_context_tokens: int = 1_048_576
    cache_device: str = "same"

    def __post_init__(self) -> None:
        if self.mode not in {"full_recompute", "stateful_truncated"}:
            raise ValueError("mode must be full_recompute or stateful_truncated")
        if self.execution_chunk_tokens < 1 or self.detach_interval_tokens < 1:
            raise ValueError("chunk and detach intervals must be positive")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if self.cache_device not in {"same", "cpu", "cuda"}:
            raise ValueError("cache_device must be same, cpu, or cuda")


@dataclass
class ChunkedForwardResult:
    loss: torch.Tensor | None
    token_count: int
    chunks: int
    boundary_tokens: int
    mode: str
    fallbacks: list[str] = field(default_factory=list)
    logits: torch.Tensor | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)


def _detach_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach()


def detach_cache(cache: KestrelCache) -> None:
    """Detach every state tensor, including pending and compressed chunks."""
    for layer in cache.layers.values():
        layer.key = _detach_tensor(layer.key)
        layer.value = _detach_tensor(layer.value)
        layer.positions = _detach_tensor(layer.positions)
        layer.pending_key = _detach_tensor(layer.pending_key)
        layer.pending_value = _detach_tensor(layer.pending_value)
        layer.pending_positions = _detach_tensor(layer.pending_positions)
        layer.compressed_key = _detach_tensor(layer.compressed_key)
        layer.compressed_value = _detach_tensor(layer.compressed_value)
        layer.compressed_positions = _detach_tensor(layer.compressed_positions)
        layer.compressed.key_chunks = [chunk.detach() for chunk in layer.compressed.key_chunks]
        layer.compressed.value_chunks = [chunk.detach() for chunk in layer.compressed.value_chunks]
        layer.compressed.position_chunks = [chunk.detach() for chunk in layer.compressed.position_chunks]
        layer.index.key_chunks = [chunk.detach() for chunk in layer.index.key_chunks]
        layer.index.scale_chunks = [
            None if chunk is None else chunk.detach() for chunk in layer.index.scale_chunks
        ]


def _sum_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    if logits.shape[1] == 0 or labels.numel() == 0:
        return None, 0
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError("logits and labels must have matching batch/sequence dimensions")
    valid = labels.ne(-100)
    count = int(valid.sum().item())
    if count == 0:
        return None, 0
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="sum",
        ignore_index=-100,
    ), count


@dataclass(frozen=True)
class _LayerCacheLayout:
    """Non-tensor description for one checkpointed cache state.

    Cache chunks are tensors, but their optional/prefix structure is ordinary
    metadata.  Keeping that structure outside the autograd graph lets the
    checkpoint boundary pass the actual K/V/index tensors explicitly without
    serializing or materializing the complete cache.
    """

    layer_idx: int
    has_local: bool
    has_pending: bool
    compressed_count: int
    index_count: int
    index_scale_flags: tuple[bool, ...]
    index_dtype: str
    local_window: int
    next_position: int


@dataclass(frozen=True)
class _CacheLayout:
    layers: tuple[_LayerCacheLayout, ...]


def _cache_to_checkpoint_state(cache: KestrelCache) -> tuple[tuple[torch.Tensor, ...], _CacheLayout]:
    """Flatten cache tensors while preserving append-only chunk boundaries."""
    tensors: list[torch.Tensor] = []
    layouts: list[_LayerCacheLayout] = []
    for layer_idx in sorted(cache.layers):
        layer = cache.layers[layer_idx]
        has_local = layer.key is not None
        has_pending = layer.pending_key is not None
        if has_local:
            if layer.value is None or layer.positions is None:
                raise RuntimeError("local cache state is incomplete")
            tensors.extend((layer.key, layer.value, layer.positions))  # type: ignore[arg-type]
        if has_pending:
            if layer.pending_value is None or layer.pending_positions is None:
                raise RuntimeError("pending cache state is incomplete")
            tensors.extend((layer.pending_key, layer.pending_value, layer.pending_positions))  # type: ignore[arg-type]
        if len(layer.compressed.key_chunks) != len(layer.compressed.value_chunks) or len(layer.compressed.key_chunks) != len(layer.compressed.position_chunks):
            raise RuntimeError("compressed cache chunk lists are misaligned")
        for key, value, positions in layer.compressed.iter_chunks():
            tensors.extend((key, value, positions))
        if len(layer.index.key_chunks) != len(layer.index.scale_chunks):
            raise RuntimeError("index cache chunk lists are misaligned")
        scale_flags = tuple(scale is not None for scale in layer.index.scale_chunks)
        for key, scale in layer.index.iter_chunks():
            tensors.append(key)
            if scale is not None:
                tensors.append(scale)
        layouts.append(
            _LayerCacheLayout(
                layer_idx=layer_idx,
                has_local=has_local,
                has_pending=has_pending,
                compressed_count=len(layer.compressed.key_chunks),
                index_count=len(layer.index.key_chunks),
                index_scale_flags=scale_flags,
                index_dtype=layer.index.dtype,
                local_window=layer.local_window,
                next_position=layer.next_position,
            )
        )
    return tuple(tensors), _CacheLayout(tuple(layouts))


def _cache_from_checkpoint_state(
    state: tuple[torch.Tensor, ...], layout: _CacheLayout | None
) -> KestrelCache:
    """Rebuild a differentiable cache object from a checkpoint boundary."""
    if layout is None:
        if state:
            raise RuntimeError("empty cache layout expected for non-empty initial state")
        return KestrelCache()
    cursor = 0
    cache = KestrelCache()
    for spec in layout.layers:
        def take() -> torch.Tensor:
            nonlocal cursor
            if cursor >= len(state):
                raise RuntimeError("checkpoint cache state ended before its layout")
            value = state[cursor]
            cursor += 1
            return value

        key = value = positions = None
        if spec.has_local:
            key, value, positions = take(), take(), take()
        pending_key = pending_value = pending_positions = None
        if spec.has_pending:
            pending_key, pending_value, pending_positions = take(), take(), take()
        compressed_keys: list[torch.Tensor] = []
        compressed_values: list[torch.Tensor] = []
        compressed_positions: list[torch.Tensor] = []
        for _ in range(spec.compressed_count):
            compressed_keys.append(take())
            compressed_values.append(take())
            compressed_positions.append(take())
        index_keys: list[torch.Tensor] = []
        index_scales: list[torch.Tensor | None] = []
        for has_scale in spec.index_scale_flags:
            index_keys.append(take())
            index_scales.append(take() if has_scale else None)
        cache.layers[spec.layer_idx] = LayerCache(
            key=key,
            value=value,
            positions=positions,
            local_window=spec.local_window,
            next_position=spec.next_position,
            pending_key=pending_key,
            pending_value=pending_value,
            pending_positions=pending_positions,
            compressed=CompressedStateStore(compressed_keys, compressed_values, compressed_positions),
            index=IndexStateStore(index_keys, index_scales, spec.index_dtype),
        )
    if cursor != len(state):
        raise RuntimeError("checkpoint cache state has tensors not described by its layout")
    return cache


def _checkpointed_chunk(
    model: torch.nn.Module,
    chunk_ids: torch.Tensor,
    state: tuple[torch.Tensor, ...],
    layout: _CacheLayout | None,
    pixel_values: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], _CacheLayout]:
    """Execute one chunk and return its new explicit cache boundary.

    ``torch.utils.checkpoint`` calls this function once during the forward
    pass and replays it while computing gradients.  The cache object is
    reconstructed from tensor state on every call, so replay cannot append to
    the already-consumed mutable cache from the original forward pass.
    """
    cache = _cache_from_checkpoint_state(state, layout)
    output = model(
        chunk_ids,
        labels=None,
        pixel_values=pixel_values,
        past_key_values=cache,
        logits_to_keep=None,
    )
    new_state, new_layout = _cache_to_checkpoint_state(cache)
    return output.logits[:, -chunk_ids.shape[1] :], new_state, new_layout


def run_chunked_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    config: LongContextConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    pixel_values: torch.Tensor | None = None,
    collect_logits: bool = False,
) -> ChunkedForwardResult:
    """Run a sequence in bounded chunks, optionally performing one update.

    The loss includes the cross-chunk next-token edge by retaining only the
    previous chunk's final logit.  Full logits are concatenated only when the
    test/debug-only ``collect_logits=True`` option is requested.
    """
    cfg = config or LongContextConfig()
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")
    if optimizer is not None and cfg.cache_device != "same":
        raise ValueError(
            "explicit compressed cache placement is inference-only; "
            "use cache_device='same' for training"
        )
    length = int(input_ids.shape[1])
    if length > cfg.max_context_tokens:
        raise ValueError(
            f"requested {length} tokens exceeds configured target {cfg.max_context_tokens}; "
            "select a larger validated context profile"
        )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    cache = KestrelCache(
        compressed_device=None if cfg.cache_device == "same" else cfg.cache_device
    )
    checkpointed = optimizer is not None and cfg.mode == "full_recompute"
    checkpoint_state: tuple[torch.Tensor, ...] = ()
    checkpoint_layout: _CacheLayout | None = None
    checkpoint_calls = 0
    loss_terms: list[tuple[torch.Tensor, int]] = []
    interval_terms: list[tuple[torch.Tensor, int]] = []
    total_loss_value = 0.0
    total_tokens = 0
    boundary_tokens = 0
    chunk_count = 0
    previous_last_logits: torch.Tensor | None = None
    collected: list[torch.Tensor] = []
    fallbacks: list[str] = []
    next_detach = cfg.detach_interval_tokens
    start = 0

    while start < length:
        if cfg.mode == "stateful_truncated":
            stop = min(start + cfg.execution_chunk_tokens, next_detach, length)
        else:
            stop = min(start + cfg.execution_chunk_tokens, length)
        chunk_ids = input_ids[:, start:stop]
        chunk_labels = labels[:, start:stop] if labels is not None else None
        chunk_pixels = pixel_values if start == 0 else None
        if checkpointed:
            # The holder is intentionally forward-local.  The non-tensor
            # layout is metadata; all differentiable cache state is returned
            # as tensors through the checkpoint boundary.
            layout_holder: dict[str, _CacheLayout] = {}

            def run_checkpointed_chunk(
                ids: torch.Tensor,
                state: tuple[torch.Tensor, ...],
                input_layout: _CacheLayout | None = checkpoint_layout,
                input_pixels: torch.Tensor | None = chunk_pixels,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                nonlocal checkpoint_calls
                checkpoint_calls += 1
                chunk_logits, next_state, next_layout = _checkpointed_chunk(
                    model, ids, state, input_layout, input_pixels
                )
                layout_holder["layout"] = next_layout
                return chunk_logits, next_state

            logits, checkpoint_state = checkpoint(
                run_checkpointed_chunk,
                chunk_ids,
                checkpoint_state,
                use_reentrant=False,
            )
            checkpoint_layout = layout_holder.get("layout")
            if checkpoint_layout is None:
                raise RuntimeError("checkpointed chunk did not return a cache layout")
            # The checkpointed helper already strips any visual prefix.
            text_logits = logits
        else:
            output = model(
                chunk_ids,
                labels=None,
                pixel_values=chunk_pixels,
                past_key_values=cache,
                logits_to_keep=None if labels is not None or collect_logits else 1,
            )
            logits = output.logits
            # A multimodal prefill has a visual prefix.  Only the final text
            # positions participate in the causal text loss.
            text_logits = logits[:, -chunk_ids.shape[1] :]
        if collect_logits:
            collected.append(text_logits.detach())

        chunk_loss: torch.Tensor | None = None
        chunk_tokens = 0
        if chunk_labels is not None:
            terms: list[torch.Tensor] = []
            if previous_last_logits is not None:
                edge_loss, edge_tokens = _sum_cross_entropy(
                    previous_last_logits.unsqueeze(1), chunk_labels[:, :1]
                )
                if edge_loss is not None:
                    terms.append(edge_loss)
                    chunk_tokens += edge_tokens
                    boundary_tokens += edge_tokens
            internal_loss, internal_tokens = _sum_cross_entropy(
                text_logits[:, :-1], chunk_labels[:, 1:]
            )
            if internal_loss is not None:
                terms.append(internal_loss)
                chunk_tokens += internal_tokens
            if terms:
                chunk_loss = torch.stack(terms).sum() / max(1, chunk_tokens)
                if cfg.mode == "full_recompute":
                    loss_terms.append((chunk_loss, chunk_tokens))
                interval_terms.append((chunk_loss, chunk_tokens))
                total_loss_value += float(chunk_loss.detach().cpu()) * chunk_tokens
                total_tokens += chunk_tokens

        previous_last_logits = text_logits[:, -1]
        chunk_count += 1
        start = stop

        if cfg.mode == "stateful_truncated" and start >= next_detach:
            if optimizer is not None and loss_terms:
                # Scale by total sequence tokens so chunks accumulate one
                # comparable gradient independent of the detach interval.
                target_tokens = max(1, int(labels[:, 1:].ne(-100).sum().item())) if labels is not None else 1
                interval_loss = sum(
                    term * (term_tokens / target_tokens) for term, term_tokens in interval_terms
                )
                interval_loss.backward()
            interval_terms = []
            detach_cache(cache)
            previous_last_logits = previous_last_logits.detach()
            next_detach += cfg.detach_interval_tokens

    if cfg.mode == "stateful_truncated" and optimizer is not None and interval_terms:
        target_tokens = max(1, int(labels[:, 1:].ne(-100).sum().item())) if labels is not None else 1
        interval_loss = sum(
            term * (term_tokens / target_tokens) for term, term_tokens in interval_terms
        )
        interval_loss.backward()

    if optimizer is not None:
        if cfg.mode == "full_recompute" and loss_terms:
            # Chunk losses are already token-normalized.  Reweight them by
            # observed token count for a true sequence-level mean.
            target_tokens = max(1, int(labels[:, 1:].ne(-100).sum().item())) if labels is not None else 1
            full_loss = sum(term * (term_tokens / target_tokens) for term, term_tokens in loss_terms)
            full_loss.backward()
        optimizer.step()

    result_loss: torch.Tensor | None = None
    if total_tokens:
        result_loss = torch.tensor(total_loss_value / total_tokens, device=input_ids.device)
    result_logits = torch.cat(collected, dim=1) if collect_logits and collected else None
    if checkpointed:
        # Rebuild only the final lightweight cache object for accounting.  It
        # shares the returned tensors and does not materialize history.
        cache = _cache_from_checkpoint_state(checkpoint_state, checkpoint_layout)
    cache_memory = cache.memory_bytes()
    return ChunkedForwardResult(
        loss=result_loss,
        token_count=total_tokens,
        chunks=chunk_count,
        boundary_tokens=boundary_tokens,
        mode=cfg.mode,
        fallbacks=fallbacks,
        logits=result_logits,
        telemetry={
            "sequence_tokens": length,
            "execution_chunk_tokens": cfg.execution_chunk_tokens,
            "detach_interval_tokens": cfg.detach_interval_tokens,
            "cache_layers": len(cache.layers),
            "full_logits_collected": collect_logits,
            "checkpointed_chunks": chunk_count if checkpointed else 0,
            "checkpoint_calls": checkpoint_calls,
            "retained_loss_graphs": 0 if checkpointed else len(loss_terms),
            "recompute_backward": checkpointed,
            "gradient_scope": "full_sequence_with_checkpointed_cache_state" if checkpointed else "retained_forward_graph",
            "cache_memory_bytes": cache_memory,
            "cache_device": cfg.cache_device,
            "evidence_label": (
                "full_gradient_training" if optimizer is not None and cfg.mode == "full_recompute"
                else "stateful_truncated_training" if optimizer is not None
                else "forward_only_stateful_truncated"
            ),
        },
    )
