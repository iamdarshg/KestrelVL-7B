"""Chunked sparse retrieval indexer; no dense T-by-T mask is materialized."""

import torch
from torch import nn


class LightningIndexer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        index_dim: int,
        topk: int,
        candidate_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.index_dim = index_dim
        self.topk = topk
        self.candidate_chunk_size = candidate_chunk_size
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        # Query heads already carry their head identity, so one projection is
        # shared per head. Keys are expanded per query head for retrieval.
        self.query = nn.Linear(head_dim, index_dim, bias=False)
        self.key = nn.Linear(head_dim, num_heads * index_dim, bias=False)
        self.scale = index_dim ** -0.5

    def project_keys(
        self,
        key_chunks,
        storage_dtype: str = "bfloat16",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Project and optionally quantize compressed keys for cache storage."""
        if storage_dtype not in {"bfloat16", "int8", "int16", "int32", "int64"}:
            raise ValueError(f"unsupported index storage dtype: {storage_dtype}")
        projected_chunks: list[torch.Tensor] = []
        scale_chunks: list[torch.Tensor | None] = []
        for full_key in key_chunks:
            if full_key.ndim != 3:
                raise ValueError("key chunks must have shape [B, M, D]")
            b, m, _ = full_key.shape
            projected = self.key(full_key).view(b, m, self.num_heads, self.index_dim).permute(0, 2, 1, 3)
            if storage_dtype == "bfloat16":
                projected_chunks.append(projected.to(torch.bfloat16))
                scale_chunks.append(None)
                continue
            dtype = getattr(torch, storage_dtype)
            qmax = float(torch.iinfo(dtype).max)
            scales = projected.float().abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / qmax
            quantized = (projected.float() / scales).round().clamp(
                float(torch.iinfo(dtype).min), qmax
            ).to(dtype)
            projected_chunks.append(quantized)
            scale_chunks.append(scales.to(torch.float16))
        return projected_chunks, scale_chunks

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep even the public eager-compatible path on the same bounded
        # implementation.  This prevents a caller from accidentally creating
        # a dense ``Q x M`` score tensor at long context.
        return self.forward_chunked(
            query,
            [key],
            query_positions,
            [key_positions],
            query_block=512,
        )

    def forward_chunked(
        self,
        query: torch.Tensor,
        key_chunks,
        query_positions: torch.Tensor,
        key_position_chunks,
        query_block: int = 512,
        projected_key_chunks=None,
        projected_key_scales=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve top-k compressed keys without a ``Q x M`` allocation.

        Each compressed chunk is scored independently and only its local
        top-k candidates are merged into a bounded global top-k heap.  Query
        blocks additionally bound the temporary score tensor, so the memory
        cost is approximately ``O(query_block * chunk_size)`` rather than
        ``O(query_length * compressed_history)``.
        """
        if query_block < 1:
            raise ValueError("query_block must be positive")
        chunks = list(key_chunks)
        positions_chunks = list(key_position_chunks)
        if len(chunks) != len(positions_chunks):
            raise ValueError("key and position chunk counts must match")
        if projected_key_chunks is not None:
            if len(projected_key_chunks) != len(chunks):
                raise ValueError("projected key chunk count must match key chunks")
            if projected_key_scales is None or len(projected_key_scales) != len(chunks):
                raise ValueError("projected key scales must match projected key chunks")
        b, h, q, _ = query.shape
        if h != self.num_heads:
            raise ValueError(f"indexer expected {self.num_heads} heads, got {h}")
        if query_positions.ndim == 1:
            query_positions = query_positions.unsqueeze(0)
        if query_positions.shape != (b, q):
            raise ValueError(f"query_positions must have shape {(b, q)}, got {tuple(query_positions.shape)}")

        output_indices: list[torch.Tensor] = []
        output_weights: list[torch.Tensor] = []
        output_valid: list[torch.Tensor] = []
        query_index = self.query(query)
        key_offset = 0
        for start in range(0, q, query_block):
            stop = min(q, start + query_block)
            block_query = query_index[:, :, start:stop]
            block_positions = query_positions[:, start:stop]
            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
            best_valid: torch.Tensor | None = None
            key_offset = 0
            for chunk_index, (full_key, full_key_positions) in enumerate(zip(chunks, positions_chunks)):
                if full_key.ndim != 3:
                    raise ValueError("key chunks must have shape [B, M, D]")
                if full_key_positions.ndim == 1:
                    full_key_positions = full_key_positions.unsqueeze(0)
                full_m = full_key.shape[1]
                if full_key_positions.shape != (b, full_m):
                    raise ValueError("key position chunk has incompatible shape")
                for substart in range(0, full_m, self.candidate_chunk_size):
                    substop = min(full_m, substart + self.candidate_chunk_size)
                    # Compressed history may intentionally reside in host
                    # RAM for 1M-token inference.  Move only this bounded
                    # candidate chunk, never the complete history.
                    key = full_key[:, substart:substop].to(
                        block_query.device, non_blocking=True
                    )
                    key_positions = full_key_positions[:, substart:substop].to(
                        block_query.device, non_blocking=True
                    )
                    m = key.shape[1]
                    if projected_key_chunks is None:
                        projected_key = self.key(key).view(b, m, h, self.index_dim).permute(0, 2, 1, 3)
                    else:
                        projected_key = projected_key_chunks[chunk_index][:, :, substart:substop]
                        scale = projected_key_scales[chunk_index]
                        if projected_key.dtype.is_floating_point:
                            projected_key = projected_key.to(
                                device=block_query.device, dtype=block_query.dtype, non_blocking=True
                            )
                        else:
                            if scale is None:
                                raise ValueError("integer projected keys require scales")
                            projected_key = (
                                projected_key.to(block_query.device, non_blocking=True).float()
                                * scale.to(block_query.device, non_blocking=True).float()
                            ).to(block_query.dtype)
                    scores = torch.einsum("bhqd,bhmd->bhqm", block_query, projected_key) * self.scale
                    # Compression groups are timestamped at their final
                    # member.  A query at that exact position is allowed to
                    # consume the group: this is the compressed equivalent
                    # of the local branch's inclusive causal comparison.
                    # Strict ``<`` would make every group-end token miss its
                    # own completed group during both prefill and decode.
                    allowed = key_positions[:, None, None, :] <= block_positions[:, None, :, None]
                    allowed = allowed.expand(-1, h, -1, -1)
                    scores = scores.masked_fill(~allowed, float("-inf"))
                    local_k = min(self.topk, m)
                    # Keep invalid candidates at -inf even when this chunk
                    # has no causal key.  Replacing them with zero locally
                    # would let an invalid candidate outrank a valid negative
                    # score from a different chunk.
                    local_scores, local_indices = scores.topk(local_k, dim=-1)
                    local_valid = allowed.gather(-1, local_indices)
                    local_indices = local_indices + key_offset + substart
                    if best_scores is None:
                        best_scores = local_scores
                        best_indices = local_indices
                        best_valid = local_valid
                    else:
                        merged_scores = torch.cat((best_scores, local_scores), dim=-1)
                        merged_indices = torch.cat((best_indices, local_indices), dim=-1)
                        merged_valid = torch.cat((best_valid, local_valid), dim=-1)
                        best_scores, selected = merged_scores.topk(min(self.topk, merged_scores.shape[-1]), dim=-1)
                        best_indices = merged_indices.gather(-1, selected)
                        best_valid = merged_valid.gather(-1, selected)
                key_offset += full_m
            if best_scores is None:
                # Keep gather-safe zero indices while reporting no candidates.
                shape = (b, h, stop - start, max(1, self.topk))
                best_scores = query.new_full(shape, float("-inf"))
                best_indices = torch.zeros(shape, dtype=torch.long, device=query.device)
                best_valid = torch.zeros(shape, dtype=torch.bool, device=query.device)
            weights = torch.softmax(best_scores.masked_fill(~best_valid, float("-inf")), dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            output_indices.append(best_indices)
            output_weights.append(weights)
            output_valid.append(best_valid)
        return (
            torch.cat(output_indices, dim=2),
            torch.cat(output_weights, dim=2),
            torch.cat(output_valid, dim=2),
        )
