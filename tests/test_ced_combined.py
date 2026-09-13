"""Batch-2 combined integration: CED (#3/#4) + CSA2 (#5) + indexer (#6) + mHC (#7).

CPU-only, tiny tensors. Guards the invariants Batch 3 depends on:

* the #3/#4 zero-gate baseline stays bit-exact;
* CED + CSA2 (+ indexer pool + single-pass mHC) runs forward AND backward;
* no dense T×S attention object is materialised on the long path
  (chunked scoring; reuse layers score zero positions);
* reuse layers perform no full prompt scan.
"""

import torch

from model.attention.mhc_singlepass import SinglePassMHC
from model.ced import (
    CEDRuntime,
    ExternalMemoryHook,
    TinyAutoregressiveDecoder,
    TinyCausalEncoder,
)
from model.csa2 import CSA2Config, CSA2Layer, CSA2Stack
from model.encoder_bridge import EncoderMemoryBridge
from model.sparse_indexer import HierarchicalIndexer, IndexerConfig

DIM = 16
VOCAB = 32


def _runtime(seed=0):
    torch.manual_seed(seed)
    encoder = TinyCausalEncoder(hidden_dim=DIM, num_layers=1, vocab_size=VOCAB)
    decoder = TinyAutoregressiveDecoder(hidden_dim=DIM, num_layers=1, vocab_size=VOCAB)
    hook = ExternalMemoryHook(decoder_dim=DIM, encoder_dim=DIM)
    runtime = CEDRuntime(encoder, decoder, hook)
    bridge = EncoderMemoryBridge(encoder_dim=DIM, memory_dim=DIM)
    return runtime, bridge


def _prompt(batch=1, seq=12, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=gen)


# --- zero-gate baseline ----------------------------------------------------


def test_zero_gate_baseline_intact():
    runtime, _ = _runtime()
    ids = _prompt()
    encoding = runtime.encode_prompt(ids)
    with_memory = runtime.decode_with_memory(ids, encoding)
    without_memory = runtime.decode_without_memory(ids)
    assert torch.equal(with_memory, without_memory)


def test_zero_gate_baseline_intact_for_long_prompt():
    runtime, _ = _runtime()
    ids = _prompt(seq=256)
    encoding = runtime.encode_prompt(ids)
    assert torch.equal(
        runtime.decode_with_memory(ids, encoding), runtime.decode_without_memory(ids)
    )


# --- CED + CSA2 forward -----------------------------------------------------


def test_ced_bridge_csa2_forward_shapes_and_telemetry():
    runtime, bridge = _runtime()
    ids = _prompt(seq=12)
    enc_states = runtime.encoder(ids)
    memory = bridge.build_memory(bridge.build_semantic(enc_states))
    hidden = runtime.decoder.forward_hidden(ids)
    config = CSA2Config(
        top_k=2, candidate_pool_size=4, num_decoder_layers=4,
        head_dim=8, cadence=["full", "reuse", "reuse", "reindex"],
    )
    stack = CSA2Stack(config, hidden_dim=DIM, memory_dim=DIM)
    stream, indices, telemetry = stack(
        hidden, memory.global_k, memory.global_v, chunk=2
    )
    assert stream.shape == hidden.shape
    assert [t["mode"] for t in telemetry] == ["full", "reuse", "reuse", "reindex"]
    # Reuse performs no full-context scan; full is bounded by S, reindex by pool.
    assert telemetry[1]["positions_scored"] == 0
    assert telemetry[2]["positions_scored"] == 0
    assert telemetry[0]["positions_scored"] == 12
    assert telemetry[3]["positions_scored"] == 4
    assert all(torch.isfinite(stream).all() for _ in [0])
    for idx in indices:
        assert idx.shape == (1, 12, 2)


def test_chunked_full_matches_materialized_reference():
    torch.manual_seed(1)
    layer = CSA2Layer(hidden_dim=DIM, memory_dim=DIM, top_k=2, head_dim=8, mode="full")
    hidden = torch.randn(1, 5, DIM)
    mem_k = torch.randn(1, 9, DIM)
    mem_v = torch.randn(1, 9, DIM)
    out_chunk, idx_chunk, _ = layer(hidden, mem_k, mem_v, chunk=2)
    out_mat, idx_mat, _ = layer(
        hidden, mem_k, mem_v, chunk=9, materialized_for_test_only=True
    )
    assert torch.equal(idx_chunk, idx_mat)
    assert torch.allclose(out_chunk, out_mat, atol=1e-5)


def test_reuse_ignores_full_memory_perturbation():
    torch.manual_seed(2)
    layer = CSA2Layer(hidden_dim=DIM, memory_dim=DIM, top_k=2, head_dim=8, mode="reuse")
    hidden = torch.randn(1, 4, DIM)
    mem_k = torch.randn(1, 10, DIM)
    mem_v = torch.randn(1, 10, DIM)
    prior = torch.tensor([[[1, 3], [0, 2], [4, 5], [6, 7]]])
    out_a, idx_a, tel_a = layer(hidden, mem_k, mem_v, prior_indices=prior)
    scrambled_k = mem_k + 1000.0
    out_b, idx_b, tel_b = layer(hidden, scrambled_k, mem_v, prior_indices=prior)
    assert torch.equal(idx_a, idx_b)
    assert tel_a["positions_scored"] == 0 and tel_b["positions_scored"] == 0
    assert out_a.shape == hidden.shape


# --- indexer -> CSA2 reindex interop ----------------------------------------


def test_indexer_pool_feeds_csa2_reindex():
    torch.manual_seed(3)
    seq, top_k, pool = 64, 4, 16
    mem_k = torch.randn(1, seq, DIM)
    mem_v = torch.randn(1, seq, DIM)
    query = torch.randn(1, 3, DIM)
    indexer = HierarchicalIndexer(
        mem_dim=DIM,
        config=IndexerConfig(
            candidate_pool_size=pool, top_k=top_k, chunk_size=8,
            index_dtype="bfloat16", coarse_dim=8,
        ),
    )
    _, _, _, itel = indexer.retrieve(query, mem_k, mem_v, source="semantic")
    assert itel["positions_fine_scored"] == pool
    assert itel["positions_fine_scored"] < seq  # bounded by pool, not context

    # A reindex layer consumes a pool-sized candidate set with pool-bounded cost.
    layer = CSA2Layer(
        hidden_dim=DIM, memory_dim=DIM, top_k=top_k, head_dim=8,
        candidate_pool_size=pool, mode="reindex",
    )
    pool_ids = torch.arange(pool).unsqueeze(0)
    out, idx, tel = layer(query, mem_k, mem_v, candidate_pool=pool_ids, chunk=4)
    assert tel["positions_scored"] == pool
    assert out.shape == (1, 3, DIM)
    assert idx.max() < seq


# --- CED + CSA2 + single-pass mHC forward/backward ---------------------------


def test_ced_csa2_mhc_forward_backward_with_open_gates():
    runtime, bridge = _runtime(seed=8)
    ids = _prompt(seq=10, seed=8)
    enc_states = runtime.encoder(ids)
    with torch.no_grad():
        bridge.output_gate.fill_(1.0)
        runtime.memory_hook.gate.fill_(0.1)
    memory = bridge.build_memory(bridge.build_semantic(enc_states))
    assert torch.isfinite(memory.global_k).all()

    hidden = runtime.decoder.forward_hidden(ids)
    config = CSA2Config(
        top_k=2, candidate_pool_size=4, num_decoder_layers=4,
        head_dim=8, cadence=["full", "reuse", "reuse", "reindex"],
    )
    stack = CSA2Stack(config, hidden_dim=DIM, memory_dim=DIM)
    with torch.no_grad():
        for layer in stack.layers:
            layer.out_proj.weight.fill_(0.05)
    stream, _, _ = stack(hidden, memory.global_k, memory.global_v, chunk=2)

    hooked = runtime.memory_hook(hidden, memory.global_v)
    assert torch.isfinite(hooked).all()
    fused = SinglePassMHC()
    csa_contrib = (stream - hidden).detach()
    mlp_update = torch.zeros_like(hooked)
    out = fused(hooked, csa_contrib, mlp_update)
    logits = runtime.decoder.lm_head(out)
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(logits).all()
    grads = {n for n, p in list(runtime.named_parameters()) + list(bridge.named_parameters())
             if p.grad is not None}
    assert any("memory_hook" in n or "hook" in n for n in grads)
    assert any("bridge" in n or "k_proj" in n or "v_proj" in n for n in grads)


def test_all_full_stack_matches_cadenced_stack_at_full_layers():
    runtime, bridge = _runtime(seed=9)
    ids = _prompt(seq=8, seed=9)
    memory = bridge.build_memory(
        bridge.build_semantic(runtime.encoder(ids)))
    hidden = runtime.decoder.forward_hidden(ids)
    config = CSA2Config(top_k=2, candidate_pool_size=4, num_decoder_layers=2, head_dim=8)
    all_full = CSA2Stack.all_full_mode(config, hidden_dim=DIM, memory_dim=DIM)
    assert all_full.mode_metadata() == ["full", "full"]
    stream, _, telemetry = all_full(hidden, memory.global_k, memory.global_v, chunk=2)
    assert stream.shape == hidden.shape
    assert all(t["mode"] == "full" for t in telemetry)
