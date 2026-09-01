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
            )
        elif self.mode == "hca":
            self.csa = HeavilyCompressedAttention(
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.hca_compression_ratio,
                config.index_head_dim,
                config.index_topk,
            )
        # Reconstruction must begin from the local branch.  sigmoid(0)=0.5
        # would inject an untrained sparse branch before distillation; -10
        # gives a genuinely near-zero opening gate while keeping it learnable.
        self.compressed_gate = nn.Parameter(torch.tensor(-10.0))

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
            cache.update(self.layer_idx, k, v, positions, compressed_key=compressed_k)
            item = cache.get(self.layer_idx)
            assert item.key is not None and item.value is not None and item.positions is not None
            key, value, key_positions = item.key, item.value, item.positions
            compressed_key = item.compressed_key if item.compressed_key is not None else key
        else:
            key, value, key_positions = k, v, positions
            compressed_key = compressed_k
        qpos = positions
        key_for_local = key.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        value_for_local = value.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        local = sliding_causal_attention(q, key_for_local, value_for_local, qpos, key_positions, self.config.sliding_window)
        if self.csa is not None:
            compressed = self.csa(compressed_q, compressed_key, value, qpos, key_positions)
            branch = local + torch.sigmoid(self.compressed_gate) * compressed
        else:
            branch = local
        return self.out(branch.transpose(1, 2)), branch
