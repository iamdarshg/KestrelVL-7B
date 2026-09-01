"""Nemotron-sized DeepSeek V4-Flash-inspired attention transplant."""

import torch
import torch.nn.functional as F
from torch import nn

from ..configuration import KestrelConfig
from .cache import KestrelCache
from .csa import CompressedSparseAttention
from .grouped_output import GroupedLowRankOutput
from .hca import HeavilyCompressedAttention
from .rope import PartialRotaryEmbedding
from .sliding import sliding_causal_attention


class V4FlashAttention(nn.Module):
    def __init__(self, config: KestrelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mode = config.layer_schedule[layer_idx]
        qdim = config.num_attention_heads * config.head_dim
        kvdim = config.num_key_value_heads * config.head_dim
        self.q_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_b = nn.Linear(config.q_lora_rank, qdim, bias=False)
        self.kv = nn.Linear(config.hidden_size, 2 * kvdim, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.out = GroupedLowRankOutput(
            config.num_attention_heads,
            config.head_dim,
            config.output_groups,
            config.output_rank,
            config.hidden_size,
        )
        self.rope = PartialRotaryEmbedding(
            config.rotary_dim, config.rope_theta, config.max_position_embeddings, config.yarn_factor
        )
        self.compress_rope = PartialRotaryEmbedding(
            config.rotary_dim,
            config.compress_rope_theta,
            config.max_position_embeddings,
            config.yarn_factor,
        )
        self.csa = None
        if self.mode == "csa":
            self.csa = CompressedSparseAttention(
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.csa_compression_ratio,
                config.index_head_dim,
                config.index_topk,
                config.candidate_chunk_size,
                config.index_dtype,
            )
        elif self.mode == "hca":
            self.csa = HeavilyCompressedAttention(
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.hca_compression_ratio,
                config.index_head_dim,
                config.index_topk,
                config.candidate_chunk_size,
                config.index_dtype,
            )
        # Reconstruction must begin exactly on the local branch.  A centered
        # sigmoid gives coefficient sigmoid(0)-0.5 == 0 while retaining a
        # non-zero derivative for recovery.  The exact-zero fast path below
        # also prevents an untrained compressed branch containing a NaN from
        # being multiplied by zero (0 * NaN is still NaN in IEEE arithmetic).
        self.compressed_gate = nn.Parameter(torch.tensor(0.0))
        # SVD is an approximation of the old GQA mixer.  Start the new
        # residual small enough to keep the frozen Nemotron body in range
        # while Muon recovers the replacement matrices from step one.
        self.output_scale = nn.Parameter(torch.tensor(config.attention_output_scale_init))

    def _positions(self, x: torch.Tensor, positions: torch.Tensor | None, cache: KestrelCache | None) -> torch.Tensor:
        if positions is not None:
            return positions
        start = cache.length(self.layer_idx) if cache is not None else 0
        return torch.arange(start, start + x.shape[1], device=x.device).view(1, -1).expand(x.shape[0], -1)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: KestrelCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self._positions(x, position_ids, cache)
        q = self.q_b(self.q_norm(self.q_a(x))).view(x.shape[0], x.shape[1], self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        kv = self.kv(x).view(x.shape[0], x.shape[1], 2, self.config.num_key_value_heads, self.config.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        raw_q = q
        raw_k = k
        q = self.rope.apply(raw_q, positions)
        k = self.rope.apply(raw_k, positions)
        compressed_q = self.compress_rope.apply(raw_q, positions)
        compressed_k = self.compress_rope.apply(raw_k, positions)
        if cache is not None:
            item = cache.get(self.layer_idx)
            if item.key is None:
                key, value, key_positions = k, v, positions
            else:
                # ``item.key`` is bounded to the local attention window, so
                # this temporary never scales with total context length.
                key = torch.cat((item.key, k), dim=2)
                value = torch.cat((item.value, v), dim=2)  # type: ignore[arg-type]
                key_positions = torch.cat((item.positions, positions), dim=1)  # type: ignore[arg-type]

            compressed_chunks: list[torch.Tensor] = []
            compressed_value_chunks: list[torch.Tensor] = []
            compressed_position_chunks: list[torch.Tensor] = []
            index_key_chunks: list[torch.Tensor] = []
            index_scale_chunks: list[torch.Tensor | None] = []
            if self.csa is not None:
                # Join only the incomplete current compression group with the
                # new token chunk.  Complete historical groups stay in the
                # append-only cache and are never concatenated here.
                pending_key = item.pending_key
                pending_value = item.pending_value
                pending_positions = item.pending_positions
                if pending_key is not None:
                    stream_key = torch.cat((pending_key, compressed_k), dim=2)
                    stream_value = torch.cat((pending_value, v), dim=2)  # type: ignore[arg-type]
                    stream_positions = torch.cat((pending_positions, positions), dim=1)  # type: ignore[arg-type]
                else:
                    stream_key, stream_value, stream_positions = compressed_k, v, positions
                new_key, new_value, new_positions, consumed = self.csa.compressor.forward_with_positions(
                    stream_key, stream_positions, value=stream_value
                )
                item.pending_key = stream_key[:, :, consumed:]
                item.pending_value = stream_value[:, :, consumed:]
                item.pending_positions = stream_positions[:, consumed:]
                if new_key.shape[2]:
                    new_index, new_scales = self.csa.indexer.project_keys(
                        [new_key.mean(dim=1)], self.csa.index_dtype
                    )
                    index_key = new_index[0]
                    index_scale = new_scales[0]
                else:
                    index_key = None
                    index_scale = None
                cache.update(
                    self.layer_idx,
                    k,
                    v,
                    positions,
                    compressed_key=new_key,
                    compressed_value=new_value,
                    compressed_positions=new_positions,
                    index_key=index_key,
                    index_scale=index_scale,
                    index_dtype=self.csa.index_dtype,
                    local_window=self.config.sliding_window,
                )
                compressed_chunks = list(item.compressed.key_chunks)
                compressed_value_chunks = list(item.compressed.value_chunks)
                compressed_position_chunks = list(item.compressed.position_chunks)
                index_key_chunks = list(item.index.key_chunks)
                index_scale_chunks = list(item.index.scale_chunks)
            else:
                cache.update(
                    self.layer_idx,
                    k,
                    v,
                    positions,
                    local_window=self.config.sliding_window,
                )
        else:
            key, value, key_positions = k, v, positions
            compressed_key = compressed_k
        qpos = positions
        key_for_local = key.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        value_for_local = value.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        local = sliding_causal_attention(q, key_for_local, value_for_local, qpos, key_positions, self.config.sliding_window)
        if self.csa is not None:
            gate = torch.sigmoid(self.compressed_gate) - 0.5
            if abs(float(gate.detach())) < 1e-8:
                branch = local
            else:
                if cache is not None:
                    compressed = self.csa.forward_from_compressed(
                        compressed_q,
                        compressed_chunks,
                        compressed_value_chunks,
                        compressed_position_chunks,
                        qpos,
                        query_block=self.config.attention_query_block,
                        index_key_chunks=index_key_chunks,
                        index_scale_chunks=index_scale_chunks,
                    )
                else:
                    compressed = self.csa(compressed_q, compressed_key, value, qpos, key_positions)
                branch = local + gate * compressed
        else:
            branch = local
        return self.output_scale * self.out(branch.transpose(1, 2)), branch
