"""Tests for issue #4: EncoderMemoryBridge + shared encoder-derived global KV.

CPU-only, tiny dims (encoder_dim=16, memory_dim=32) for speed.
"""

from __future__ import annotations

import pytest
import torch

from model.encoder_bridge import (
    EncoderMemory,
    EncoderMemoryBridge,
    LayerMemoryView,
)

ENC = 16
MEM = 32


def _bridge(**kwargs) -> EncoderMemoryBridge:
    kwargs.setdefault("encoder_dim", ENC)
    kwargs.setdefault("memory_dim", MEM)
    return EncoderMemoryBridge(**kwargs)


def _states(batch=2, seq=5, dim=ENC, seed=0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32)


# --- gating -----------------------------------------------------------------


def test_output_gate_init_exactly_zero():
    bridge = _bridge()
    assert float(bridge.output_gate.detach().cpu()) == 0.0


def test_zero_gate_equivalence():
    bridge = _bridge()
    sem = _states()
    mem_on = bridge(sem, gate_enabled=True)
    mem_off = bridge(sem, gate_enabled=False)
    assert torch.equal(mem_off.global_k, torch.zeros_like(mem_off.global_k))
    assert torch.equal(mem_off.global_v, torch.zeros_like(mem_off.global_v))
    # Gate inits to 0.0, so enabled path is also exactly zero at init.
    assert torch.equal(mem_on.global_k, mem_off.global_k)
    assert torch.equal(mem_on.global_v, mem_off.global_v)
    blank = bridge.blank_memory_like(mem_on)
    assert torch.equal(blank.global_k, mem_off.global_k)
    assert torch.equal(blank.global_v, mem_off.global_v)


def test_gate_scales_contribution_when_opened():
    bridge = _bridge()
    with torch.no_grad():
        bridge.output_gate.fill_(1.0)
    mem = bridge(_states())
    assert torch.count_nonzero(mem.global_k).item() > 0
    assert torch.count_nonzero(mem.global_v).item() > 0


# --- shapes / dtype / device -------------------------------------------------


def test_forward_shapes_and_tags():
    bridge = _bridge()
    mem = bridge(_states(batch=2, seq=7))
    assert mem.global_k.shape == (2, 7, MEM)
    assert mem.global_v.shape == (2, 7, MEM)
    assert mem.seq_len == 7
    assert mem.memory_dim == MEM
    assert mem.dtype == str(torch.float32)
    assert mem.device == "cpu"


def test_dtype_device_coverage():
    bridge = EncoderMemoryBridge(encoder_dim=ENC, memory_dim=MEM, dtype=torch.float64, device="cpu")
    mem = bridge(_states().to(torch.float32))  # forward casts inputs
    assert mem.global_k.dtype == torch.float64
    assert mem.dtype == str(torch.float64)
    assert mem.device == "cpu"


def test_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        EncoderMemoryBridge(encoder_dim=0, memory_dim=MEM)
    with pytest.raises(ValueError):
        EncoderMemoryBridge(encoder_dim=ENC, memory_dim=-1)
    with pytest.raises(ValueError):
        EncoderMemoryBridge(encoder_dim=ENC, memory_dim=MEM, detail_dim=0)


def test_rejects_non_finite_inputs():
    bridge = _bridge()
    bad = _states()
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        bridge(bad)
    bad2 = torch.full((1, 2, ENC), float("nan"))
    with pytest.raises(ValueError):
        bridge.build_semantic(bad2)


# --- determinism / reuse / one-time construction ------------------------------


def test_deterministic_prefill_decode_reuse():
    bridge = _bridge()
    a = bridge(_states(seed=11))
    b = bridge(_states(seed=11))
    assert a.fingerprint == b.fingerprint
    assert torch.equal(a.global_k, b.global_k)
    assert torch.equal(a.global_v, b.global_v)


def test_reuse_for_layers_shares_storage():
    bridge = _bridge()
    mem = bridge(_states())
    n = 4
    views = bridge.reuse_for_layers(mem, n)
    assert len(views) == n
    assert all(isinstance(v, LayerMemoryView) for v in views)
    for v in views:
        assert v.k.data_ptr() == mem.global_k.data_ptr()
        assert v.v.data_ptr() == mem.global_v.data_ptr()
        assert v.k.shape == mem.global_k.shape
        assert v.v.shape == mem.global_v.shape
    with pytest.raises(ValueError):
        bridge.reuse_for_layers(mem, 0)


def test_one_time_construction_no_accumulation():
    bridge = _bridge()
    before = set(bridge.__dict__)
    m1 = bridge(_states(seed=3))
    mid = set(bridge.__dict__)
    m2 = bridge(_states(seed=3))
    after = set(bridge.__dict__)
    assert before == mid == after  # no per-token history retained
    assert m1.fingerprint == m2.fingerprint
    assert not hasattr(bridge, "_history")


# --- dual taps / ablation ------------------------------------------------------


def test_detail_semantic_ablation():
    bridge = _bridge()
    sem_proj = bridge.build_semantic(_states(seed=5))
    assert sem_proj.shape == (2, 5, MEM)
    mem_sem_only = bridge.build_memory(sem_proj, None)
    assert mem_sem_only.source_tap == "semantic"
    assert mem_sem_only.global_k.shape == (2, 5, MEM)
    det_proj = bridge.build_detail(_states(seed=6))
    mem_both = bridge.build_memory(sem_proj, det_proj)
    assert mem_both.source_tap == "detail"
    assert mem_both.global_k.shape == (2, 5, MEM)
    # Detail with a different sequence length still fuses.
    det_long = bridge.build_detail(torch.randn(2, 9, ENC))
    mem_long = bridge.build_memory(sem_proj, det_long)
    assert mem_long.global_k.shape == (2, 5, MEM)
    # Detail actually changes the fused memory once the gate is open.
    with torch.no_grad():
        bridge.output_gate.fill_(1.0)
    fused_only = bridge.build_memory(sem_proj, None)
    fused_both = bridge.build_memory(sem_proj, det_proj)
    assert not torch.equal(fused_only.global_k, fused_both.global_k)


def test_forward_accepts_raw_detail_states():
    bridge = _bridge(detail_dim=ENC)
    mem = bridge(_states(seed=1), _states(seed=2))
    assert mem.source_tap == "detail"
    assert mem.global_k.shape == (2, 5, MEM)


# --- serialization --------------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    bridge = _bridge()
    mem = bridge(_states(seed=9))
    d = mem.to_dict()
    assert isinstance(d["global_k"], list)
    back = EncoderMemory.from_dict(d)
    assert back.fingerprint == mem.fingerprint
    assert torch.equal(back.global_k, mem.global_k)
    assert torch.equal(back.global_v, mem.global_v)
    assert back.source_tap == mem.source_tap


def test_from_dict_tamper_raises():
    bridge = _bridge()
    d = bridge(_states(seed=9)).to_dict()
    d["global_k"][0][0][0] += 1.0
    with pytest.raises(ValueError):
        EncoderMemory.from_dict(d)
    d2 = bridge(_states(seed=9)).to_dict()
    d2["fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        EncoderMemory.from_dict(d2)


def test_snapshot_strict_load_round_trip():
    bridge = _bridge()
    state = bridge.state_dict_snapshot()
    meta = bridge.metadata()
    assert {"encoder_dim", "memory_dim", "dtype", "device", "fingerprint"} <= set(meta)
    other = _bridge()
    other.load_snapshot_strict(state, meta)
    for (k, v), (k2, v2) in zip(sorted(state.items()), sorted(other.state_dict_snapshot().items())):
        assert k == k2
        assert torch.equal(v.cpu(), v2.cpu())
    # Same input now yields identical fingerprints across the two bridges.
    x = _states(seed=21)
    assert bridge(x).fingerprint == other(x).fingerprint


def test_snapshot_metadata_mismatch_raises():
    bridge = _bridge()
    state, meta = bridge.state_dict_snapshot(), bridge.metadata()
    bad = dict(meta)
    bad["memory_dim"] = MEM + 1
    with pytest.raises(ValueError):
        bridge.load_snapshot_strict(state, bad)


def test_strict_load_missing_key_raises_without_reinit():
    bridge = _bridge()
    state, meta = bridge.state_dict_snapshot(), bridge.metadata()
    before = {k: v.clone() for k, v in state.items()}
    dropped = {k: v for k, v in state.items() if k != sorted(state)[0]}
    with pytest.raises((RuntimeError, ValueError)):
        bridge.load_snapshot_strict(dropped, meta)
    # Failed strict load leaves params untouched (no random reinit).
    after = bridge.state_dict_snapshot()
    for k in before:
        assert torch.equal(before[k].cpu(), after[k].cpu())


def test_no_random_reinit_on_strict_load():
    bridge = _bridge()
    state, meta = bridge.state_dict_snapshot(), bridge.metadata()
    with torch.no_grad():
        for p in bridge.parameters():
            p.add_(1.0)
    bridge.load_snapshot_strict(state, meta)
    restored = bridge.state_dict_snapshot()
    for k in state:
        assert torch.equal(state[k].cpu(), restored[k].cpu())


# --- freeze / backward ------------------------------------------------------------


def test_trainable_parameter_counts_and_freeze():
    bridge = _bridge()
    counts = bridge.trainable_parameter_counts()
    assert counts["trainable"] == counts["total"] > 0
    bridge.freeze()
    assert bridge.trainable_parameter_counts()["trainable"] == 0
    assert all(not p.requires_grad for p in bridge.parameters())
    bridge.unfreeze()
    assert bridge.trainable_parameter_counts()["trainable"] == counts["total"]
    assert all(p.requires_grad for p in bridge.parameters())


def test_backward_flows_unfrozen_blocked_frozen():
    bridge = _bridge()
    with torch.no_grad():
        bridge.output_gate.fill_(1.0)
    x = _states(seed=4)
    mem = bridge(x)
    (mem.global_k.sum() + mem.global_v.sum()).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in bridge.parameters())
    assert bridge.k_proj.weight.grad is not None
    assert torch.count_nonzero(bridge.k_proj.weight.grad).item() > 0

    frozen = _bridge()
    with torch.no_grad():
        frozen.output_gate.fill_(1.0)
    frozen.freeze()
    mem_f = frozen(x)
    # Fully frozen: no grad graph exists, so backward is blocked outright.
    with pytest.raises(RuntimeError):
        (mem_f.global_k.sum() + mem_f.global_v.sum()).backward()
    assert all(p.grad is None for p in frozen.parameters())


# --- splice rejection / memory interfaces ------------------------------------------


def test_rejects_direct_hidden_state_splicing():
    bridge = _bridge()
    with pytest.raises(RuntimeError):
        bridge.splice_into_decoder_states(_states(), torch.randn(2, 5, MEM))


def test_dense_reference_and_sparse_stub():
    bridge = _bridge()
    mem = bridge(_states(seed=8))
    k, v = bridge.dense_reference(mem)
    assert torch.equal(k, mem.global_k)
    assert torch.equal(v, mem.global_v)
    idx, ks, vs = bridge.sparse_memory_stub(mem, top_k=2)
    assert idx.shape == (2, 2)
    assert ks.shape == (2, 2, MEM)
    assert vs.shape == (2, 2, MEM)
    with pytest.raises(ValueError):
        bridge.sparse_memory_stub(mem, top_k=6)  # seq_len is 5


def test_blank_memory_like_is_exactly_zero():
    bridge = _bridge()
    with torch.no_grad():
        bridge.output_gate.fill_(1.0)
    mem = bridge(_states(seed=8))
    blank = bridge.blank_memory_like(mem)
    assert torch.equal(blank.global_k, torch.zeros_like(mem.global_k))
    assert torch.equal(blank.global_v, torch.zeros_like(mem.global_v))
    assert blank.seq_len == mem.seq_len
    assert blank.memory_dim == mem.memory_dim
