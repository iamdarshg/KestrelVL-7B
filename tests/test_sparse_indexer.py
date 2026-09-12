"""Tests for issue #6: two-stage hierarchical sparse indexer.

CPU-only, tiny dims (S<=64, pool<=32, top_k<=8) for speed.
"""

from __future__ import annotations

import pytest
import torch

from model.sparse_indexer import (
    CoarseIndex,
    HierarchicalIndexer,
    IndexerConfig,
    dense_reference_topk,
    recall_at_k,
    span_recall,
    synthetic_copy_task,
)

MEM_DIM = 16
SEQ = 32
QT = 4


def _cfg(**kwargs):
    kwargs.setdefault("candidate_pool_size", 32)
    kwargs.setdefault("top_k", 4)
    kwargs.setdefault("chunk_size", 8)
    kwargs.setdefault("index_dtype", "bfloat16")
    kwargs.setdefault("coarse_dim", 8)
    return IndexerConfig(**kwargs)


def _tensors(batch=2, seq=SEQ, t=QT, dim=MEM_DIM, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mem_k = torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32)
    mem_v = torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32)
    query = torch.randn(batch, t, dim, generator=gen, dtype=torch.float32)
    return mem_k, mem_v, query


# --- config -----------------------------------------------------------------


def test_config_defaults():
    cfg = IndexerConfig()
    assert cfg.candidate_pool_size == 2048
    assert cfg.top_k == 256
    assert cfg.chunk_size == 512
    assert cfg.index_dtype == "bfloat16"
    assert cfg.coarse_dim == 64
    assert cfg.deterministic is True
    cfg.validate()


def test_config_validate_rejects():
    with pytest.raises(ValueError):
        IndexerConfig(top_k=0).validate()
    with pytest.raises(ValueError):
        IndexerConfig(candidate_pool_size=4, top_k=8).validate()
    with pytest.raises(ValueError):
        IndexerConfig(index_dtype="fp8").validate()
    with pytest.raises(ValueError):
        IndexerConfig(chunk_size=0).validate()
    with pytest.raises(ValueError):
        IndexerConfig(coarse_dim=0).validate()


# --- exact agreement with dense reference ------------------------------------


def test_exact_agreement_with_dense_reference():
    # Pool covers the full sequence, so the fine stage sees every position
    # and must match the exact dense reference.
    mem_k, mem_v, query = _tensors()
    indexer = HierarchicalIndexer(MEM_DIM, config=_cfg())
    k_sel, v_sel, indices, tel = indexer.retrieve(query, mem_k, mem_v)
    _, ref = dense_reference_topk(query, mem_k, 4)
    assert torch.equal(indices, ref)
    assert k_sel.shape == (2, QT, 4, MEM_DIM)
    assert v_sel.shape == (2, QT, 4, MEM_DIM)
    # Gathered values match the indexed positions.
    b = torch.arange(2)[:, None, None]
    assert torch.equal(k_sel, mem_k[b, indices])
    assert torch.equal(v_sel, mem_v[b, indices])
    assert tel["recall_vs_dense"] == pytest.approx(1.0)


def test_recall_vs_dense_via_dense_indices():
    mem_k, mem_v, query = _tensors()
    indexer = HierarchicalIndexer(MEM_DIM, config=_cfg())
    _, ref = dense_reference_topk(query, mem_k, 4)
    _, _, _, tel = indexer.retrieve(query, mem_k, mem_v, dense_indices=ref)
    assert tel["recall_vs_dense"] == pytest.approx(1.0)


def test_chunked_vs_reference_agreement():
    mem_k, mem_v, query = _tensors(seq=48)
    ref_cfg = _cfg(candidate_pool_size=48, top_k=6, chunk_size=64)
    tiny_cfg = _cfg(candidate_pool_size=48, top_k=6, chunk_size=3)
    torch.manual_seed(0)
    a = HierarchicalIndexer(MEM_DIM, config=ref_cfg)
    torch.manual_seed(0)
    b = HierarchicalIndexer(MEM_DIM, config=tiny_cfg)
    b.load_snapshot_strict(a.state_dict_snapshot(), a.metadata())
    _, _, idx_a, _ = a.retrieve(query, mem_k, mem_v)
    _, _, idx_b, _ = b.retrieve(query, mem_k, mem_v)
    assert torch.equal(idx_a, idx_b)
    _, ref = dense_reference_topk(query, mem_k, 6)
    assert torch.equal(idx_a, ref)


# --- determinism --------------------------------------------------------------


def test_determinism_identical_inputs():
    mem_k, mem_v, query = _tensors()
    indexer = HierarchicalIndexer(MEM_DIM, config=_cfg())
    out1 = indexer.retrieve(query, mem_k, mem_v)
    out2 = indexer.retrieve(query, mem_k, mem_v)
    for x, y in zip(out1[:3], out2[:3]):
        assert torch.equal(x, y)


# --- candidate bound + positions accounting -----------------------------------


def test_candidate_bound_enforced():
    mem_k, mem_v, query = _tensors(seq=64)
    cfg = _cfg(candidate_pool_size=16, top_k=8, chunk_size=5)
    indexer = HierarchicalIndexer(MEM_DIM, config=cfg)
    _, _, indices, tel = indexer.retrieve(query, mem_k, mem_v)
    assert indices.shape == (2, QT, 8)
    assert tel["candidate_pool_size"] == 16
    assert tel["top_k"] == 8
    # Fine stage never scores more than the bounded pool.
    assert tel["positions_fine_scored"] <= 16
    assert tel["positions_coarse_scored"] == 64


def test_no_full_qs_materialization_outside_reference():
    # Chunked scoring must account every position while only ever
    # materializing at most chunk_size columns at once.
    mem_k, mem_v, query = _tensors(seq=40)
    cfg = _cfg(candidate_pool_size=40, top_k=4, chunk_size=6)
    indexer = HierarchicalIndexer(MEM_DIM, config=cfg)
    _, _, _, tel = indexer.retrieve(query, mem_k, mem_v)
    assert tel["positions_coarse_scored"] == 40
    assert tel["positions_fine_scored"] == 40
    assert tel["chunk_size"] == 6
    assert tel["chunk_size"] < tel["positions_coarse_scored"]


# --- dtype modes ---------------------------------------------------------------


def test_int8_and_bf16_modes():
    mem_k, mem_v, query = _tensors()
    bf16 = HierarchicalIndexer(MEM_DIM, config=_cfg(index_dtype="bfloat16"))
    int8 = HierarchicalIndexer(MEM_DIM, config=_cfg(index_dtype="int8"))
    _, _, idx_b, tel_b = bf16.retrieve(query, mem_k, mem_v)
    _, _, idx_i, tel_i = int8.retrieve(query, mem_k, mem_v)
    assert idx_b.shape == idx_i.shape == (2, QT, 4)
    assert tel_b["dtype_mode"] == "bfloat16"
    assert tel_i["dtype_mode"] == "int8"
    assert tel_i["index_bytes_per_token"] < tel_b["index_bytes_per_token"]
    index_b = bf16.build_coarse_index(mem_k)
    index_i = int8.build_coarse_index(mem_k)
    assert isinstance(index_b, CoarseIndex)
    assert index_b.index_bytes_per_token() == tel_b["index_bytes_per_token"]
    assert index_i.index_bytes_per_token() == tel_i["index_bytes_per_token"]
    assert index_i.codes.dtype == torch.int8


# --- recall metric --------------------------------------------------------------


def test_recall_at_k_correct():
    ref = torch.tensor([[[1, 2, 3, 4]]])
    assert recall_at_k(ref.clone(), ref) == pytest.approx(1.0)
    disjoint = torch.tensor([[[5, 6, 7, 8]]])
    assert recall_at_k(disjoint, ref) == pytest.approx(0.0)
    partial = torch.tensor([[[1, 2, 5, 6]]])
    assert recall_at_k(partial, ref) == pytest.approx(0.5)
    # Averaged over batch/query entries.
    mixed = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    two = torch.tensor([[[1, 2, 3, 4]], [[1, 2, 3, 4]]])
    assert recall_at_k(mixed, two) == pytest.approx(0.5)


# --- sources ---------------------------------------------------------------------


def test_semantic_and_detail_sources_recorded():
    mem_k, mem_v, query = _tensors()
    indexer = HierarchicalIndexer(MEM_DIM, config=_cfg())
    for source in ("semantic", "detail"):
        _, _, _, tel = indexer.retrieve(query, mem_k, mem_v, source=source)
        assert tel["source"] == source
    with pytest.raises(ValueError):
        indexer.retrieve(query, mem_k, mem_v, source="nope")


# --- serialization ------------------------------------------------------------------


def test_serialization_roundtrip_resume_identical():
    mem_k, mem_v, query = _tensors()
    indexer = HierarchicalIndexer(MEM_DIM, config=_cfg())
    _, _, idx_before, _ = indexer.retrieve(query, mem_k, mem_v)
    state = indexer.state_dict_snapshot()
    meta = indexer.metadata()
    clone = HierarchicalIndexer(MEM_DIM, config=_cfg())
    with torch.no_grad():
        clone.mem_projector.weight.add_(1.0)
    clone.load_snapshot_strict(state, meta)
    _, _, idx_after, _ = clone.retrieve(query, mem_k, mem_v)
    assert torch.equal(idx_before, idx_after)
    bad = dict(meta)
    bad["coarse_dim"] = -1
    with pytest.raises(ValueError):
        clone.load_snapshot_strict(state, bad)


# --- synthetic copy task ---------------------------------------------------------------


def test_synthetic_copy_task_recall_one():
    mem_k, mem_v, query, span = synthetic_copy_task(seq_len=16, span=(4, 8))
    assert span == (4, 8)
    indexer = HierarchicalIndexer(mem_k.shape[-1], config=_cfg(candidate_pool_size=16, top_k=4))
    _, _, indices, tel = indexer.retrieve(query, mem_k, mem_v, compute_recall_on_tiny=True)
    assert span_recall(indices, span) == pytest.approx(1.0)
    assert tel["recall_vs_dense"] == pytest.approx(1.0)


def test_span_recall_partial_and_zero():
    indices = torch.tensor([[[4, 5, 9, 10]]])
    assert span_recall(indices, (4, 6)) == pytest.approx(1.0)
    assert span_recall(indices, (4, 8)) == pytest.approx(0.5)
    assert span_recall(indices, (0, 2)) == pytest.approx(0.0)


# --- far-context: no positional bias ------------------------------------------------------


def test_no_positional_bias_documented():
    import model.sparse_indexer as mod

    assert "positional" in mod.__doc__.lower()
    assert "no positional bias" in mod.__doc__.lower()
