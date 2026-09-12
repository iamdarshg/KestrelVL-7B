"""Issue #6: two-stage hierarchical sparse indexer.

Coarse candidate pool -> fine Top-K retrieval over plain-tensor K/V memory.

Far-context check support: no positional bias in scoring -- every score is a
pure content-based dot product (coarse projected codes, fine full-dim keys)
multiplied by an optional learned temperature scalar. No positional encodings,
alibi slopes, or distance penalties are added anywhere, so far-context tokens
compete on content alone.

Duck-typing: all memory inputs are plain ``torch.Tensor`` (``[B, S, D]``);
this module never imports the encoder bridge or ``csa2``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn

_VALID_DTYPES = ("bfloat16", "int8")
_VALID_SOURCES = ("semantic", "detail")
_TINY_AUTO_RECALL_POSITIONS = 8192
_TINY_FORCED_RECALL_POSITIONS = 65536


@dataclass
class IndexerConfig:
    """Knobs for :class:`HierarchicalIndexer`."""

    candidate_pool_size: int = 2048
    top_k: int = 256
    chunk_size: int = 512
    index_dtype: str = "bfloat16"
    coarse_dim: int = 64
    deterministic: bool = True

    def validate(self) -> "IndexerConfig":
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError(f"top_k must be an int, got {self.top_k!r}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.candidate_pool_size < self.top_k:
            raise ValueError(
                "candidate_pool_size must be >= top_k, got "
                f"{self.candidate_pool_size} < {self.top_k}"
            )
        if self.index_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"index_dtype must be one of {_VALID_DTYPES}, got {self.index_dtype!r}"
            )
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise ValueError(f"chunk_size must be an int, got {self.chunk_size!r}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.coarse_dim < 1:
            raise ValueError(f"coarse_dim must be >= 1, got {self.coarse_dim}")
        return self


@dataclass
class CoarseIndex:
    """Compact per-token codes produced by ``build_coarse_index``."""

    codes: torch.Tensor  # bf16 [B,S,C] or int8 [B,S,C]
    scales: torch.Tensor | None  # bf16 [B,S] (int8 mode only)
    zero_points: torch.Tensor | None  # bf16 [B,S] (int8 mode only)
    mode: str
    seq_len: int
    coarse_dim: int

    def index_bytes_per_token(self) -> float:
        if self.mode == "bfloat16":
            return float(self.coarse_dim * 2)
        if self.mode == "int8":
            # 1 byte/code + amortized bf16 scale + bf16 zero-point per token.
            return float(self.coarse_dim + 2 + 2)
        raise ValueError(f"unknown index mode {self.mode!r}")

    def dequantized_codes(self) -> torch.Tensor:
        """Return float32 codes ``[B, S, C]`` (dequantized in int8 mode)."""
        if self.mode == "bfloat16":
            return self.codes.float()
        if self.mode == "int8":
            if self.scales is None or self.zero_points is None:
                raise ValueError("int8 index is missing scale/zero-point")
            return (self.codes.float() + 128.0) * self.scales.float().unsqueeze(
                -1
            ) + self.zero_points.float().unsqueeze(-1)
        raise ValueError(f"unknown index mode {self.mode!r}")


def _deterministic_topk(
    scores: torch.Tensor, indices: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k with score-desc / index-asc tie-break (deterministic)."""
    order_idx = torch.argsort(indices, dim=-1, stable=True)
    sorted_scores = torch.gather(scores, -1, order_idx)
    sorted_indices = torch.gather(indices, -1, order_idx)
    order_score = torch.argsort(sorted_scores, dim=-1, descending=True, stable=True)
    top = order_score[..., :k]
    return torch.gather(sorted_scores, -1, top), torch.gather(sorted_indices, -1, top)


def dense_reference_topk(
    query: torch.Tensor, mem_k: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """REFERENCE ONLY: exact full softmax-score Top-K (never use for retrieval).

    Materializes the full ``[B, T, S]`` score tensor, so this is only suitable
    for tiny-input correctness checks. Softmax is monotonic, hence the Top-K
    matches the raw dot-product Top-K; ties break by ascending index.
    """
    if query.dim() != 3 or mem_k.dim() != 3:
        raise ValueError("query and mem_k must be [B, T, Q] / [B, S, D] tensors")
    if query.shape[0] != mem_k.shape[0]:
        raise ValueError("batch mismatch between query and mem_k")
    if query.shape[-1] != mem_k.shape[-1]:
        raise ValueError("query dim must equal memory dim for dense reference")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    k = min(top_k, mem_k.shape[1])
    scores = torch.matmul(query.float(), mem_k.float().transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    all_idx = torch.arange(mem_k.shape[1], device=query.device).expand(
        probs.shape[0], probs.shape[1], mem_k.shape[1]
    )
    return _deterministic_topk(probs, all_idx, k)


def recall_at_k(retrieved_indices: torch.Tensor, reference_indices: torch.Tensor) -> float:
    """Fraction of reference ids recovered, averaged over batch/query entries."""
    ref = reference_indices.long()
    ret = retrieved_indices.long()
    if ref.dim() < 1 or ret.dim() < 1 or ref.shape[-1] == 0:
        raise ValueError("indices must be non-empty along the last dim")
    k = ref.shape[-1]
    recalls = []
    for b in range(ref.shape[0]):
        rows_r = (
            ret[b].reshape(-1, ret.shape[-1]) if ret.dim() > 1 else ret.reshape(-1, ret.shape[-1])
        )
        rows_f = ref[b].reshape(-1, k) if ref.dim() > 1 else ref.reshape(-1, k)
        for row_r, row_f in zip(rows_r, rows_f):
            hits = len(set(row_r.tolist()) & set(row_f.tolist()))
            recalls.append(hits / k)
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def span_recall(indices: torch.Tensor, span: tuple[int, int] | list[int]) -> float:
    """Fraction of ``[start, end)`` span positions covered by retrieved indices."""
    start, end = int(span[0]), int(span[1])
    if not (0 <= start < end):
        raise ValueError(f"span must satisfy 0 <= start < end, got {span!r}")
    hits = ((indices >= start) & (indices < end)).sum(dim=-1).float()
    coverage = (hits / float(end - start)).clamp(max=1.0)
    return float(coverage.mean().item())


def synthetic_copy_task(
    seq_len: int, span: tuple[int, int] | list[int], mem_dim: int = 16, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Tiny memory + query with a KNOWN supporting span (supervision helper).

    Span rows share one strong pattern vector; the query equals that pattern,
    so exact Top-K with ``top_k >= len(span)`` recovers the span (recall 1.0).
    No training loop lives here -- this only builds retrieval targets.
    """
    start, end = int(span[0]), int(span[1])
    if not (0 <= start < end <= seq_len):
        raise ValueError(f"span {span!r} out of bounds for seq_len={seq_len}")
    if mem_dim < 1:
        raise ValueError(f"mem_dim must be >= 1, got {mem_dim}")
    gen = torch.Generator().manual_seed(seed)
    pattern = torch.randn(mem_dim, generator=gen, dtype=torch.float32)
    pattern = pattern / pattern.norm().clamp_min(1e-6) * 6.0
    mem_k = torch.randn(1, seq_len, mem_dim, generator=gen, dtype=torch.float32) * 0.5
    noise = torch.randn(end - start, mem_dim, generator=gen, dtype=torch.float32)
    mem_k[0, start:end, :] = pattern.unsqueeze(0) + noise * 0.01
    mem_v = torch.randn(1, seq_len, mem_dim, generator=gen, dtype=torch.float32) * 0.5
    query = pattern.view(1, 1, mem_dim).clone()
    return mem_k, mem_v, query, (start, end)


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


class HierarchicalIndexer(nn.Module):
    """Two-stage coarse-pool -> fine Top-K indexer (no randomness, no dropout)."""

    def __init__(
        self,
        mem_dim: int,
        query_dim: int | None = None,
        config: IndexerConfig | None = None,
    ) -> None:
        super().__init__()
        self.mem_dim = _require_positive_int("mem_dim", mem_dim)
        self.query_dim = _require_positive_int(
            "query_dim", query_dim if query_dim is not None else mem_dim
        )
        self.config = config.validate() if config is not None else IndexerConfig()
        self.config.validate()
        self.mem_projector = nn.Linear(self.mem_dim, self.config.coarse_dim)
        self.query_projector = nn.Linear(self.query_dim, self.config.coarse_dim)
        # Learned temperature scalar; positive init preserves score ordering.
        self.temperature = nn.Parameter(torch.tensor(1.0))

    # -- coarse index ------------------------------------------------------
    def build_coarse_index(self, mem: torch.Tensor) -> CoarseIndex:
        if not isinstance(mem, torch.Tensor) or mem.dim() != 3:
            raise ValueError("mem must be a [B, S, D] tensor")
        if mem.shape[-1] != self.mem_dim:
            raise ValueError(f"mem last dim must be {self.mem_dim}, got {mem.shape[-1]}")
        full = self.mem_projector(mem.float())
        mode = self.config.index_dtype
        seq_len = mem.shape[1]
        if mode == "bfloat16":
            return CoarseIndex(
                codes=full.to(torch.bfloat16),
                scales=None,
                zero_points=None,
                mode=mode,
                seq_len=seq_len,
                coarse_dim=self.config.coarse_dim,
            )
        if mode == "int8":
            mins = full.amin(dim=-1, keepdim=True)
            maxs = full.amax(dim=-1, keepdim=True)
            spans = (maxs - mins).masked_fill(maxs == mins, 1.0)
            scale = spans / 255.0
            quant = torch.round((full - mins) / scale).clamp(0, 255).to(torch.int16)
            codes = (quant - 128).to(torch.int8)
            return CoarseIndex(
                codes=codes,
                scales=scale.squeeze(-1).to(torch.bfloat16),
                zero_points=mins.squeeze(-1).to(torch.bfloat16),
                mode=mode,
                seq_len=seq_len,
                coarse_dim=self.config.coarse_dim,
            )
        raise ValueError(f"unknown index_dtype {mode!r}")

    # -- chunked exact top-k over a [B, T, N] score stream ------------------
    def _chunked_topk(
        self,
        score_fn,
        n: int,
        k: int,
        batch: int,
        queries: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.config.chunk_size
        best_scores = torch.empty(batch, queries, 0, device=device)
        best_idx = torch.empty(batch, queries, 0, dtype=torch.long, device=device)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            scores = score_fn(start, stop)
            width = stop - start
            new_idx = torch.arange(start, stop, device=device).expand(batch, queries, width)
            cand_scores = torch.cat([best_scores, scores], dim=-1)
            cand_idx = torch.cat([best_idx, new_idx], dim=-1)
            keep = min(k, cand_scores.shape[-1])
            best_scores, best_idx = _deterministic_topk(cand_scores, cand_idx, keep)
        return best_scores, best_idx

    # -- retrieve -----------------------------------------------------------
    def retrieve(
        self,
        query: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        source: str = "semantic",
        dense_indices: torch.Tensor | None = None,
        compute_recall_on_tiny: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if source not in _VALID_SOURCES:
            raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
        for name, tensor in (("query", query), ("mem_k", mem_k), ("mem_v", mem_v)):
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
                raise ValueError(f"{name} must be a 3D [B, T/S, D] tensor")
        if not (query.shape[0] == mem_k.shape[0] == mem_v.shape[0]):
            raise ValueError("batch mismatch between query, mem_k, mem_v")
        if mem_k.shape != mem_v.shape:
            raise ValueError("mem_k and mem_v shapes must match")
        if query.shape[-1] != self.query_dim:
            raise ValueError(f"query last dim must be {self.query_dim}, got {query.shape[-1]}")
        if mem_k.shape[-1] != self.mem_dim:
            raise ValueError(f"mem last dim must be {self.mem_dim}, got {mem_k.shape[-1]}")
        if self.query_dim != self.mem_dim:
            raise ValueError(
                "fine stage needs query_dim == mem_dim for exact content dot "
                f"products, got {self.query_dim} vs {self.mem_dim}"
            )
        batch, num_queries, _ = query.shape
        seq_len = mem_k.shape[1]
        if seq_len < 1:
            raise ValueError("mem sequence length must be >= 1")

        pool_size = min(self.config.candidate_pool_size, seq_len)
        top_k = min(self.config.top_k, pool_size)
        temp = self.temperature.to(dtype=torch.float32)

        index = self.build_coarse_index(mem_k)
        mem_codes = index.dequantized_codes()
        query_codes = self.query_projector(query.float())

        def coarse_scores(start: int, stop: int) -> torch.Tensor:
            chunk_codes = mem_codes[:, start:stop, :].float()
            return torch.einsum("btc,bwc->btw", query_codes, chunk_codes) * temp

        _, candidates = self._chunked_topk(
            coarse_scores, seq_len, pool_size, batch, num_queries, query.device
        )

        batch_idx = torch.arange(batch, device=query.device)[:, None, None]
        pool_k = mem_k[batch_idx, candidates]
        pool_v = mem_v[batch_idx, candidates]
        flat_pool_k = pool_k.reshape(batch * num_queries, pool_size, -1)
        flat_q_full = query.float().reshape(batch * num_queries, 1, -1)

        def fine_scores(start: int, stop: int) -> torch.Tensor:
            block = flat_pool_k[:, start:stop, :]
            return (
                torch.matmul(flat_q_full, block.transpose(-1, -2)).view(batch, num_queries, -1)
                * temp
            )

        _, top_in_pool = self._chunked_topk(
            fine_scores, pool_size, top_k, batch, num_queries, query.device
        )
        indices = torch.gather(candidates, -1, top_in_pool)
        batch_idx = torch.arange(batch, device=query.device)[:, None, None]
        query_idx = torch.arange(num_queries, device=query.device)[None, :, None]
        k_sel = pool_k[batch_idx, query_idx, top_in_pool]
        v_sel = pool_v[batch_idx, query_idx, top_in_pool]

        total_positions = batch * num_queries * seq_len
        recall: float | None = None
        if dense_indices is not None:
            recall = recall_at_k(indices.detach().cpu(), dense_indices.detach().cpu())
        elif compute_recall_on_tiny:
            if total_positions <= _TINY_FORCED_RECALL_POSITIONS:
                _, ref = dense_reference_topk(query.detach(), mem_k.detach(), top_k)
                recall = recall_at_k(indices.detach().cpu(), ref.detach().cpu())
        elif total_positions <= _TINY_AUTO_RECALL_POSITIONS:
            _, ref = dense_reference_topk(query.detach(), mem_k.detach(), top_k)
            recall = recall_at_k(indices.detach().cpu(), ref.detach().cpu())

        telemetry = {
            "candidate_pool_size": pool_size,
            "top_k": top_k,
            "index_bytes_per_token": index.index_bytes_per_token(),
            "positions_coarse_scored": seq_len,
            "positions_fine_scored": pool_size,
            "recall_vs_dense": recall,
            "source": source,
            "dtype_mode": self.config.index_dtype,
            "chunk_size": self.config.chunk_size,
            "coarse_dim": self.config.coarse_dim,
        }
        return k_sel, v_sel, indices, telemetry

    # -- serialization -------------------------------------------------------
    def state_dict_snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def metadata(self) -> dict:
        digest = hashlib.sha256()
        digest.update(f"{self.mem_dim}|{self.query_dim}|{self.config.coarse_dim}|".encode())
        digest.update(f"{self.config.index_dtype}|{self.config.candidate_pool_size}|".encode())
        digest.update(f"{self.config.top_k}|{self.config.chunk_size}".encode())
        for key in sorted(self.state_dict()):
            tensor = self.state_dict()[key].detach().to("cpu", torch.float32)
            digest.update(tensor.contiguous().numpy().tobytes())
        return {
            "mem_dim": self.mem_dim,
            "query_dim": self.query_dim,
            "coarse_dim": self.config.coarse_dim,
            "index_dtype": self.config.index_dtype,
            "candidate_pool_size": self.config.candidate_pool_size,
            "top_k": self.config.top_k,
            "chunk_size": self.config.chunk_size,
            "deterministic": self.config.deterministic,
            "fingerprint": digest.hexdigest(),
        }

    def load_snapshot_strict(self, state: dict[str, torch.Tensor], metadata: dict) -> None:
        expected = {
            "mem_dim": self.mem_dim,
            "query_dim": self.query_dim,
            "coarse_dim": self.config.coarse_dim,
            "index_dtype": self.config.index_dtype,
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"snapshot metadata mismatch for {key!r}: got "
                    f"{metadata.get(key)!r}, expected {want!r}"
                )
        self.load_state_dict(state, strict=True)
