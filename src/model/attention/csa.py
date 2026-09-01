"""Compressed Sparse Attention (CSA) reference branch."""

import torch
from torch import nn

from .compressor import LearnedCompressor
from .lightning_indexer import LightningIndexer


class CompressedSparseAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        kv_heads: int,
        head_dim: int,
        ratio: int,
        index_dim: int,
        topk: int,
        candidate_chunk_size: int = 64,
        index_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.ratio = heads, kv_heads, head_dim, ratio
        self.index_dtype = index_dtype
        self.compressor = LearnedCompressor(head_dim, ratio)
        self.indexer = LightningIndexer(head_dim, heads, index_dim, topk, candidate_chunk_size)
        self.sink_key = nn.Parameter(torch.zeros(heads, head_dim))
        self.sink_value = nn.Parameter(torch.zeros(heads, head_dim))
        self.sink_logit = nn.Parameter(torch.zeros(heads))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        # K/V are [B, KV, T, D]; query is [B, H, Q, D].
        ck, cv, cp, _ = self.compressor.forward_with_positions(key, key_positions, value=value)
        if ck.shape[2] == 0:
            return self.sink_value.view(1, self.heads, 1, self.head_dim).expand(
                query.shape[0], -1, query.shape[2], -1
            )
        return self.forward_from_compressed(query, [ck], [cv], [cp], query_positions)

    def forward_from_compressed(
        self,
        query: torch.Tensor,
        key_chunks: list[torch.Tensor],
        value_chunks: list[torch.Tensor],
        position_chunks: list[torch.Tensor],
        query_positions: torch.Tensor,
        query_block: int = 512,
        index_key_chunks: list[torch.Tensor] | None = None,
        index_scale_chunks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """CSA over append-only compressed state with bounded temporaries."""
        if not (len(key_chunks) == len(value_chunks) == len(position_chunks)):
            raise ValueError("compressed key/value/position chunk counts must match")
        if not key_chunks:
            return self.sink_value.view(1, self.heads, 1, self.head_dim).expand(
                query.shape[0], -1, query.shape[2], -1
            )
        # Retrieval uses a shared compressed key so the index remains one
        # compact structure.  Values retain KV-head identity: with two KV
        # heads each query head gathers its corresponding compressed value
        # stream.  This makes the intermediate KV-head ablation meaningful
        # instead of silently discarding every stream except stream zero.
        retrieval_chunks = [chunk.mean(dim=1) for chunk in key_chunks]
        if index_key_chunks is not None and len(index_key_chunks) == len(key_chunks):
            retrieval_index_chunks = index_key_chunks
            retrieval_index_scales = index_scale_chunks
        else:
            retrieval_index_chunks, retrieval_index_scales = self.indexer.project_keys(
                retrieval_chunks, self.index_dtype
            )
        query_to_kv = torch.arange(self.heads, device=query.device) * self.kv_heads // self.heads
        outputs: list[torch.Tensor] = []
        for start in range(0, query.shape[2], query_block):
            stop = min(query.shape[2], start + query_block)
            indices, weights, selected_valid = self.indexer.forward_chunked(
                query[:, :, start:stop],
                retrieval_chunks,
                query_positions[:, start:stop],
                position_chunks,
                query_block=stop - start,
                projected_key_chunks=retrieval_index_chunks,
                projected_key_scales=retrieval_index_scales,
            )
            block_out = query.new_zeros(query.shape[0], self.heads, stop - start, self.head_dim)
            offset = 0
            for compressed_key, compressed_value, _ in zip(key_chunks, value_chunks, position_chunks):
                count = compressed_key.shape[2]
                if count == 0:
                    continue
                selected_values = compressed_value.index_select(1, query_to_kv)
                in_chunk = (indices >= offset) & (indices < offset + count) & selected_valid
                safe_indices = (indices - offset).clamp(0, count - 1)
                gathered = selected_values[:, :, None, :, :].expand(
                    query.shape[0], self.heads, stop - start, count, self.head_dim
                ).gather(3, safe_indices[..., None].expand(-1, -1, -1, -1, self.head_dim))
                block_out = block_out + (gathered * weights[..., None] * in_chunk[..., None]).sum(dim=3)
                offset += count
            has_valid = selected_valid.any(dim=-1, keepdim=False)
            sink = self.sink_value[None, :, None, :].expand_as(block_out)
            outputs.append(torch.where(has_valid[..., None], block_out, sink))
        return torch.cat(outputs, dim=2)

    def _compressed_positions(self, positions: torch.Tensor) -> torch.Tensor:
        groups = torch.unique_consecutive(positions[0] // self.ratio)
        complete = []
        for g in groups.tolist():
            if (positions[0] // self.ratio == g).sum().item() == self.ratio:
                complete.append((g + 1) * self.ratio - 1)
        return positions.new_tensor(complete)
