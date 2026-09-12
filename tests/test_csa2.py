"""Tests for issue #5: CSA2 Full / Reindex / Reuse sparse attention (CPU, tiny)."""

from __future__ import annotations

import pytest
import torch

from model.csa2 import (
    CSA2Config,
    CSA2Layer,
    CSA2Stack,
    LayerMode,
    ReuseWithoutPriorError,
    dense_reference_attention,
    topk_deterministic,
)

H = 8
D = 8
HD = 8


def _tensors(b=2, t=4, s=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(b, t, H, generator=gen, dtype=torch.float32)
    gen2 = torch.Generator().manual_seed(seed + 1)
    mem_k = torch.randn(b, s, D, generator=gen2, dtype=torch.float32)
    gen3 = torch.Generator().manual_seed(seed + 2)
    mem_v = torch.randn(b, s, D, generator=gen3, dtype=torch.float32)
    return hidden, mem_k, mem_v


def _eye_layer(mode="full", top_k=2, pool_size=4):
    layer = CSA2Layer(
        hidden_dim=H,
        memory_dim=D,
        top_k=top_k,
        head_dim=HD,
        candidate_pool_size=pool_size,
        mode=mode,
    )
    with torch.no_grad():
        layer.q_proj.weight.copy_(torch.eye(HD, H))
        layer.k_proj.weight.copy_(torch.eye(HD, D))
    return layer


# --- topk_deterministic -------------------------------------------------------


def test_topk_deterministic_values_and_tie_break():
    scores = torch.tensor([[0.5, 0.9, 0.9, 0.1]])
    vals, idx = topk_deterministic(scores, 2)
    assert idx.tolist() == [[1, 2]]
    assert torch.allclose(vals, torch.tensor([[0.9, 0.9]]))
    vals2, idx2 = topk_deterministic(scores, 2)
    assert torch.equal(idx, idx2)


def test_topk_deterministic_desc_order():
    scores = torch.tensor([[1.0, 3.0, 2.0]])
    vals, idx = topk_deterministic(scores, 3)
    assert idx.tolist() == [[1, 2, 0]]
    assert torch.allclose(vals, torch.tensor([[3.0, 2.0, 1.0]]))


# --- config / cadence ----------------------------------------------------------


def test_layer_mode_cyclic_and_validate():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["full", "reuse", "reuse", "reindex"],
        num_decoder_layers=6,
        head_dim=HD,
    )
    assert cfg.layer_mode(0) == "full"
    assert cfg.layer_mode(4) == "full"
    assert cfg.layer_mode(5) == "reuse"
    cfg.validate()


def test_bad_cadence_raises():
    with pytest.raises(ValueError):
        CSA2Config(
            top_k=2,
            candidate_pool_size=4,
            cadence=[],
            num_decoder_layers=2,
            head_dim=HD,
        )
    with pytest.raises(ValueError):
        CSA2Config(
            top_k=2,
            candidate_pool_size=4,
            cadence=["full", "bogus"],
            num_decoder_layers=2,
            head_dim=HD,
        )
    with pytest.raises(ValueError):
        CSA2Config(
            top_k=0,
            candidate_pool_size=4,
            cadence=["full"],
            num_decoder_layers=2,
            head_dim=HD,
        )


def test_cadence_configurability_respected():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=2,
        cadence=["full", "reuse"],
        num_decoder_layers=4,
        head_dim=HD,
    )
    stack = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    assert stack.mode_metadata() == ["full", "reuse", "full", "reuse"]


def test_all_full_mode():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["full", "reuse", "reindex"],
        num_decoder_layers=3,
        head_dim=HD,
    )
    stack = CSA2Stack.all_full_mode(cfg, hidden_dim=H, memory_dim=D)
    assert stack.mode_metadata() == ["full", "full", "full"]
    hidden, mem_k, mem_v = _tensors()
    out, indices, tels = stack(hidden, mem_k, mem_v)
    assert out.shape == hidden.shape
    assert all(t["mode"] == "full" for t in tels)


def test_layer_mode_enum_values():
    assert LayerMode.FULL.value == "full"
    assert LayerMode.REINDEX.value == "reindex"
    assert LayerMode.REUSE.value == "reuse"


# --- full vs dense reference ----------------------------------------------------


def test_full_vs_dense_reference_agreement():
    b, t, s = 1, 3, 5
    hidden, mem_k, mem_v = _tensors(b, t, s, seed=7)
    layer = _eye_layer(mode="full", top_k=s)
    with torch.no_grad():
        layer.out_proj.weight.copy_(torch.eye(H, D))
    out, indices, tel = layer(hidden, mem_k, mem_v, chunk=2, contribution_enabled=True)
    ref = dense_reference_attention(hidden, mem_k, mem_v)
    assert torch.allclose(out, ref, atol=1e-5)
    assert tel["positions_scored"] == s


def test_chunked_vs_materialized_agreement():
    hidden, mem_k, mem_v = _tensors()
    layer = _eye_layer(mode="full", top_k=2)
    out_c, idx_c, _ = layer(hidden, mem_k, mem_v, chunk=2)
    out_m, idx_m, _ = layer(hidden, mem_k, mem_v, chunk=8, materialized_for_test_only=True)
    assert torch.equal(idx_c, idx_m)
    assert torch.equal(out_c, out_m)


def test_determinism_same_input_identical_indices():
    hidden, mem_k, mem_v = _tensors(seed=3)
    layer = _eye_layer(mode="full", top_k=2)
    _, idx1, _ = layer(hidden, mem_k, mem_v, chunk=2)
    _, idx2, _ = layer(hidden, mem_k, mem_v, chunk=2)
    assert torch.equal(idx1, idx2)


# --- reindex ---------------------------------------------------------------------


def test_reindex_restricted_to_pool():
    hidden, mem_k, mem_v = _tensors()
    pool = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    layer = _eye_layer(mode="reindex", top_k=2)
    out, indices, tel = layer(hidden, mem_k, mem_v, candidate_pool=pool, chunk=2)
    assert tel["positions_scored"] <= pool.shape[1]
    assert tel["positions_scored"] == pool.shape[1]
    for b in range(2):
        allowed = set(pool[b].tolist())
        for row in indices[b].tolist():
            assert set(row) <= allowed
    assert out.shape == hidden.shape


def test_reindex_without_pool_raises():
    hidden, mem_k, mem_v = _tensors()
    layer = _eye_layer(mode="reindex", top_k=2)
    with pytest.raises(ValueError):
        layer(hidden, mem_k, mem_v)


# --- reuse ------------------------------------------------------------------------


def test_reuse_consumes_exact_prior_with_zero_scoring():
    hidden, mem_k, mem_v = _tensors()
    prior = torch.tensor([[[0, 1]] * 4, [[6, 7]] * 4])
    layer = _eye_layer(mode="reuse", top_k=2)
    out, indices, tel = layer(hidden, mem_k, mem_v, prior_indices=prior)
    assert torch.equal(indices, prior)
    assert tel["positions_scored"] == 0
    assert out.shape == hidden.shape


def test_reuse_without_prior_raises():
    hidden, mem_k, mem_v = _tensors()
    layer = _eye_layer(mode="reuse", top_k=2)
    with pytest.raises(ReuseWithoutPriorError):
        layer(hidden, mem_k, mem_v)


def test_stack_reuse_without_prior_raises():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["reuse"],
        num_decoder_layers=2,
        head_dim=HD,
    )
    stack = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    hidden, mem_k, mem_v = _tensors()
    with pytest.raises(ReuseWithoutPriorError):
        stack(hidden, mem_k, mem_v)


def test_stack_threads_full_reindex_reuse():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["full", "reindex", "reuse"],
        num_decoder_layers=3,
        head_dim=HD,
    )
    stack = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    hidden, mem_k, mem_v = _tensors()
    out, indices, tels = stack(hidden, mem_k, mem_v, chunk=2)
    assert out.shape == hidden.shape
    assert [t["mode"] for t in tels] == ["full", "reindex", "reuse"]
    assert tels[0]["positions_scored"] == mem_k.shape[1]
    assert tels[1]["positions_scored"] <= cfg.candidate_pool_size
    assert tels[2]["positions_scored"] == 0
    assert torch.equal(indices[2], indices[1])


# --- causal / zero-init / telemetry -------------------------------------------------


def test_causal_rule_past_independent_of_future():
    hidden, mem_k, mem_v = _tensors()
    layer = _eye_layer(mode="full", top_k=2)
    out1, _, _ = layer(hidden, mem_k, mem_v, chunk=2)
    future_perturbed = hidden.clone()
    future_perturbed[:, 2:, :] += 25.0
    out2, _, _ = layer(future_perturbed, mem_k, mem_v, chunk=2)
    assert torch.equal(out1[:, :2, :], out2[:, :2, :])


def test_zero_init_contribution():
    hidden, mem_k, mem_v = _tensors()
    layer = CSA2Layer(hidden_dim=H, memory_dim=D, top_k=2, head_dim=HD, mode="full")
    out, _, _ = layer(hidden, mem_k, mem_v, chunk=2)
    assert torch.equal(out, torch.zeros_like(out))
    out_off, _, _ = layer(hidden, mem_k, mem_v, chunk=2, contribution_enabled=False)
    assert torch.equal(out_off, torch.zeros_like(out_off))


def test_telemetry_fields():
    hidden, mem_k, mem_v = _tensors()
    layer = _eye_layer(mode="full", top_k=2, pool_size=4)
    _, _, tel = layer(hidden, mem_k, mem_v, chunk=2)
    for key in (
        "mode",
        "positions_scored",
        "top_k",
        "candidate_pool_size",
        "index_bytes_per_token",
    ):
        assert key in tel
    assert tel["mode"] == "full"
    assert tel["top_k"] == 2
    assert tel["index_bytes_per_token"] == 2 * 8


def test_no_txt_allocation_positions_scored_accounting():
    hidden, mem_k, mem_v = _tensors(t=4, s=8)
    full = _eye_layer(mode="full", top_k=2)
    _, _, tel_f = full(hidden, mem_k, mem_v, chunk=2)
    assert tel_f["positions_scored"] == 8
    pool = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    reidx = _eye_layer(mode="reindex", top_k=2)
    _, _, tel_r = reidx(hidden, mem_k, mem_v, candidate_pool=pool, chunk=2)
    assert tel_r["positions_scored"] == 4
    reuse = _eye_layer(mode="reuse", top_k=2)
    prior = torch.zeros(2, 4, 2, dtype=torch.long)
    _, _, tel_u = reuse(hidden, mem_k, mem_v, prior_indices=prior)
    assert tel_u["positions_scored"] == 0


# --- serialization ------------------------------------------------------------------


def test_serialization_round_trip_and_resume():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["full", "reuse"],
        num_decoder_layers=2,
        head_dim=HD,
    )
    stack = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    hidden, mem_k, mem_v = _tensors()
    out1, idx1, _ = stack(hidden, mem_k, mem_v, chunk=2)
    snap = stack.state_dict_snapshot()
    meta = stack.metadata()
    assert meta["cadence"] == ["full", "reuse"]
    assert meta["top_k"] == 2
    assert "modes" in meta
    stack2 = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    stack2.load_snapshot_strict(snap, meta)
    assert stack2.mode_metadata() == ["full", "reuse"]
    out2, idx2, tels2 = stack2(hidden, mem_k, mem_v, chunk=2)
    assert torch.equal(out1, out2)
    for a, b in zip(idx1, idx2):
        assert torch.equal(a, b)
    assert tels2[1]["mode"] == "reuse"
    assert tels2[1]["positions_scored"] == 0


def test_snapshot_metadata_mismatch_raises():
    cfg = CSA2Config(
        top_k=2,
        candidate_pool_size=4,
        cadence=["full", "reuse"],
        num_decoder_layers=2,
        head_dim=HD,
    )
    stack = CSA2Stack(cfg, hidden_dim=H, memory_dim=D)
    snap, meta = stack.state_dict_snapshot(), stack.metadata()
    bad = dict(meta)
    bad["cadence"] = ["full", "full"]
    with pytest.raises(ValueError):
        stack.load_snapshot_strict(snap, bad)
