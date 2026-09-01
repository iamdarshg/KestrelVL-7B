import pytest
import torch

from model import KestrelConfig, KestrelForCausalLM
from model.attention.cache import KestrelCache
from model.attention.csa import CompressedSparseAttention
from model.attention.grouped_output import GroupedLowRankOutput
from model.attention.lightning_indexer import LightningIndexer
from model.attention.mhc import ManifoldHyperConnection
from model.attention.rope import PartialRotaryEmbedding
from model.attention.sliding import sliding_causal_attention
from model.nemotron import RealNemotronKestrelForCausalLM
from model.vision.internvit import InternViTEncoder, _extract_token_sequence
from model.vision.projector import AdaptiveVisionProjector
from release.runtime import Q4Linear, load_q4_runtime
from release.serialization import load_q4_bundle, save_q4_bundle
from training.long_context import LongContextConfig, run_chunked_forward
from model.attention.module import V4FlashAttention
from model.transplant.svd_init import initialize_attention_from_dense
from eval.long_context import estimate_cache_memory


def positions(length: int) -> torch.Tensor:
    return torch.arange(length).view(1, -1)


def test_sliding_causal_attention_has_no_future_leakage() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    out = sliding_causal_attention(q, k, v, positions(6), positions(6), 3)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 4:] += 100
    v2[:, :, 4:] += 100
    changed = sliding_causal_attention(q, k2, v2, positions(6), positions(6), 3)
    assert torch.allclose(out[:, :, :4], changed[:, :, :4], atol=1e-6)


@pytest.mark.parametrize("length", [1024, 4096, 16384])
def test_sliding_lengths_are_finite_without_dense_mask(length: int) -> None:
    q = torch.randn(1, 1, length, 4)
    k = torch.randn(1, 1, length, 4)
    v = torch.randn(1, 1, length, 4)
    out = sliding_causal_attention(q, k, v, positions(length), positions(length), 16, query_block=128)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_compression_does_not_emit_incomplete_future_group() -> None:
    module = CompressedSparseAttention(2, 1, 8, ratio=4, index_dim=4, topk=4)
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 1, 8, 8)
    v = torch.randn(1, 1, 8, 8)
    out = module(q, k, v, positions(8), positions(8))
    k_changed, v_changed = k.clone(), v.clone()
    k_changed[:, :, 4:] += 50
    v_changed[:, :, 4:] += 50
    changed = module(q, k_changed, v_changed, positions(8), positions(8))
    # Queries in positions 0..3 only see the sink; queries 4..7 may see group 0.
    assert torch.allclose(out[:, :, :4], changed[:, :, :4], atol=1e-6)
    assert torch.isfinite(out).all()


def test_compressed_group_is_visible_at_its_inclusive_causal_endpoint() -> None:
    torch.manual_seed(101)
    module = CompressedSparseAttention(1, 1, 4, ratio=2, index_dim=2, topk=1)
    with torch.no_grad():
        module.compressor.key_mix.weight.copy_(torch.eye(4))
        module.compressor.value_mix.weight.copy_(torch.eye(4))
        module.sink_value.zero_()
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 2, 4)
    value = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]]])
    output = module(query, key, value, positions(2), positions(2))
    # The compressed group is timestamped at position 1.  Position 0 still
    # sees only the sink; position 1 must see the completed group itself.
    assert torch.allclose(output[0, 0, 0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(output[0, 0, 1], value[0, 0].mean(dim=0), atol=1e-6)


def test_compression_preserves_multiple_kv_streams() -> None:
    torch.manual_seed(11)
    module = CompressedSparseAttention(4, 2, 8, ratio=2, index_dim=4, topk=4)
    q = torch.randn(1, 4, 8, 8)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    base = module(q, k, v, positions(8), positions(8))
    altered = v.clone()
    altered[:, 1] += 3.0
    changed = module(q, k, altered, positions(8), positions(8))
    assert not torch.allclose(base[:, 2:], changed[:, 2:])


def test_indexer_topk_is_deterministic_and_bounded() -> None:
    torch.manual_seed(2)
    indexer = LightningIndexer(8, 2, 4, topk=64)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 5, 8)
    result_a = indexer(q, k, positions(3), positions(5))
    result_b = indexer(q, k, positions(3), positions(5))
    assert result_a[0].shape[-1] == 5
    assert torch.equal(result_a[0], result_b[0])
    assert torch.isfinite(result_a[1]).all()


def test_partial_rope_only_changes_requested_subdimension() -> None:
    rope = PartialRotaryEmbedding(4, 10000, 32)
    x = torch.randn(1, 2, 3, 8)
    out = rope.apply(x, positions(3))
    assert out.shape == x.shape
    assert torch.equal(out[..., 4:], x[..., 4:])


def test_mhc_is_doubly_stochastic_and_finite() -> None:
    mhc = ManifoldHyperConnection(streams=2, sinkhorn_iters=10)
    matrix = mhc.matrix()
    assert torch.allclose(matrix.sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(matrix.sum(-2), torch.ones(2), atol=1e-5)
    base, update = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    assert torch.isfinite(mhc(base, update)).all()


def test_grouped_output_preserves_cross_group_o_projection_components() -> None:
    torch.manual_seed(102)
    config = KestrelConfig.tiny(use_vision=False, output_rank=32)
    attention = V4FlashAttention(config, layer_idx=2)
    wq = torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size)
    wk = torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
    wv = torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
    wo = torch.randn(config.hidden_size, config.hidden_size)
    errors = initialize_attention_from_dense(attention, wq, wk, wv, wo, old_kv_heads=1)
    assert attention.out.up.shape == (
        config.output_groups,
        config.output_rank,
        config.hidden_size,
    )
    # Rank 32 is sufficient for each 32-by-64 input-group slice in this tiny
    # config, so the grouped factorization should reconstruct the complete
    # dense projection, including its off-diagonal output-group components.
    assert errors["o_relative_error"] < 1e-5


def test_model_prefill_decode_cache_equivalence() -> None:
    torch.manual_seed(3)
    config = KestrelConfig.tiny(use_vision=False)
    model = KestrelForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        full = model(ids).logits
        cache = KestrelCache()
        pieces = [model(ids[:, :1], past_key_values=cache).logits]
        for i in range(1, ids.shape[1]):
            pieces.append(model(ids[:, i : i + 1], past_key_values=cache).logits)
        decoded = torch.cat(pieces, dim=1)
    assert torch.allclose(full, decoded, atol=2e-5, rtol=2e-5)


def test_gradient_finite() -> None:
    config = KestrelConfig.tiny(use_vision=False)
    model = KestrelForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (1, 16))
    loss = model(ids, labels=ids).loss
    assert loss is not None
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_cache_keeps_bounded_local_state_and_chunked_compressed_state() -> None:
    torch.manual_seed(12)
    config = KestrelConfig.tiny(use_vision=False, sliding_window=4)
    model = KestrelForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 40))
    cache = KestrelCache()
    with torch.no_grad():
        model(ids, past_key_values=cache)
    assert cache.length() == 40
    assert cache.get(0).key is not None and cache.get(0).key.shape[2] == 4
    assert cache.get(2).compressed_token_count == 10
    assert cache.get(3).compressed_token_count == 5
    assert len(cache.get(2).compressed.key_chunks) == 1
    assert cache.get(2).index.dtype == "int8"
    assert cache.get(2).index.token_count == 10
    assert cache.get(2).memory_bytes["index_state"] > 0

    restored = KestrelCache.from_state_dict(cache.state_dict())
    assert restored.length() == 40
    assert restored.get(0).key is not None and restored.get(0).key.shape[2] == 4
    assert restored.get(2).compressed_token_count == 10
    assert restored.get(2).index.token_count == 10


def test_compressed_cache_can_live_on_cpu_for_long_context_inference() -> None:
    torch.manual_seed(122)
    config = KestrelConfig.tiny(use_vision=False, sliding_window=4)
    model = KestrelForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 24))
    cache = KestrelCache(compressed_device="cpu")
    with torch.inference_mode():
        offloaded = model(ids, past_key_values=cache).logits
        for layer_index in (2, 3):
            layer = cache.get(layer_index)
            assert layer.compressed.key_chunks
            assert all(chunk.device.type == "cpu" for chunk in layer.compressed.key_chunks)
            assert all(chunk.device.type == "cpu" for chunk in layer.compressed.value_chunks)
            assert all(chunk.device.type == "cpu" for chunk in layer.index.key_chunks)
        expected = model(ids).logits
    assert torch.allclose(expected, offloaded, atol=2e-5, rtol=2e-5)


def test_bfloat16_index_state_is_reused_without_scales() -> None:
    torch.manual_seed(121)
    config = KestrelConfig.tiny(use_vision=False, index_dtype="bfloat16", sliding_window=4)
    model = KestrelForCausalLM(config).eval()
    cache = KestrelCache()
    ids = torch.randint(0, config.vocab_size, (1, 24))
    with torch.inference_mode():
        model(ids, past_key_values=cache)
    index = cache.get(2).index
    assert index.dtype == "bfloat16"
    assert index.key_chunks and all(chunk.dtype == torch.bfloat16 for chunk in index.key_chunks)
    assert all(scale is None for scale in index.scale_chunks)
    assert index.memory_bytes > 0


def test_chunked_indexer_matches_eager_topk() -> None:
    torch.manual_seed(13)
    indexer = LightningIndexer(8, 2, 4, topk=4, candidate_chunk_size=2)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 7, 8)
    qpos, kpos = positions(5), positions(7)
    eager = indexer(query, key, qpos, kpos)
    chunked = indexer.forward_chunked(
        query,
        [key[:, :3], key[:, 3:]],
        qpos,
        [kpos[:, :3], kpos[:, 3:]],
        query_block=2,
    )
    assert torch.equal(eager[0], chunked[0])
    assert torch.allclose(eager[1], chunked[1], atol=1e-6)
    assert torch.equal(eager[2], chunked[2])


def test_chunked_csa_matches_single_compressed_chunk() -> None:
    torch.manual_seed(14)
    module = CompressedSparseAttention(2, 1, 8, ratio=2, index_dim=4, topk=4, candidate_chunk_size=2)
    query = torch.randn(1, 2, 8, 8)
    key = torch.randn(1, 1, 8, 8)
    value = torch.randn(1, 1, 8, 8)
    ck, cv, cp, _ = module.compressor.forward_with_positions(key, positions(8), value=value)
    eager = module.forward_from_compressed(query, [ck], [cv], [cp], positions(8))
    split = module.forward_from_compressed(
        query,
        [ck[:, :, :2], ck[:, :, 2:]],
        [cv[:, :, :2], cv[:, :, 2:]],
        [cp[:, :2], cp[:, 2:]],
        positions(8),
    )
    assert torch.allclose(eager, split, atol=1e-6, rtol=1e-6)


def test_long_context_chunked_forward_does_not_need_full_logit_buffer() -> None:
    torch.manual_seed(15)
    config = KestrelConfig.tiny(use_vision=False, sliding_window=32)
    model = KestrelForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 24))
    with torch.no_grad():
        expected = model(ids).logits
        result = run_chunked_forward(
            model,
            ids,
            config=LongContextConfig(mode="full_recompute", execution_chunk_tokens=5, max_context_tokens=64),
            collect_logits=True,
        )
    assert result.logits is not None
    assert torch.allclose(expected, result.logits, atol=2e-5, rtol=2e-5)
    assert result.telemetry["full_logits_collected"] is True


def test_chunked_forward_reports_explicit_cpu_cache_policy() -> None:
    torch.manual_seed(151)
    config = KestrelConfig.tiny(use_vision=False, sliding_window=8)
    model = KestrelForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 24))
    with torch.inference_mode():
        result = run_chunked_forward(
            model,
            ids,
            config=LongContextConfig(
                mode="stateful_truncated",
                execution_chunk_tokens=5,
                detach_interval_tokens=10,
                max_context_tokens=64,
                cache_device="cpu",
            ),
            collect_logits=True,
        )
    assert result.telemetry["cache_device"] == "cpu"
    assert result.telemetry["evidence_label"] == "forward_only_stateful_truncated"
    assert result.logits is not None and result.logits.shape == (1, 24, config.vocab_size)


def test_cache_memory_estimator_is_monotonic_and_does_not_allocate_context() -> None:
    config = KestrelConfig(use_vision=False)
    short = estimate_cache_memory(config, 4096)
    long = estimate_cache_memory(config, 1_048_576)
    assert long["bytes"]["total"] > short["bytes"]["total"]
    assert long["context_tokens"] == 1_048_576
    assert long["includes_model_weights"] is False
    assert long["includes_peak_activations"] is False


def test_q4_bundle_is_pickle_free_and_round_trips(tmp_path) -> None:
    torch.manual_seed(16)
    state = {
        "layer.weight": torch.randn(3, 129, dtype=torch.float32),
        "norm.weight": torch.randn(3, dtype=torch.bfloat16),
    }
    output = save_q4_bundle(state, tmp_path / "release", {"model_type": "kestrel"})
    restored = load_q4_bundle(output)
    assert restored["norm.weight"].dtype == torch.bfloat16
    assert restored["layer.weight"].shape == state["layer.weight"].shape
    assert torch.allclose(restored["layer.weight"].float(), state["layer.weight"], atol=0.3)


def test_q4_runtime_keeps_linear_weights_packed(tmp_path) -> None:
    torch.manual_seed(161)
    config = KestrelConfig.tiny(use_vision=False, num_hidden_layers=2, layer_schedule=["sliding", "csa"])
    source = KestrelForCausalLM(config).eval()
    release = save_q4_bundle(source.state_dict(), tmp_path / "q4", config.to_dict())

    restored = KestrelForCausalLM(config).eval()
    load_q4_runtime(restored, release, device="cpu")
    quantized_linears = [module for module in restored.modules() if isinstance(module, Q4Linear)]
    assert quantized_linears
    assert all(module.packed_weight.dtype == torch.uint8 for module in quantized_linears)
    ids = torch.randint(0, config.vocab_size, (1, 9))
    with torch.inference_mode():
        output = restored(ids).logits
    assert output.shape == (1, 9, config.vocab_size)
    assert torch.isfinite(output).all()


def test_vision_policy_offloads_and_caches_encoded_tokens() -> None:
    torch.manual_seed(17)
    config = KestrelConfig.tiny(
        use_vision=True,
        vision_offload_threshold=0,
        vision_budget_ordinary=4,
    )
    model = KestrelForCausalLM(config).eval()
    pixels = torch.rand(3, 28, 28)
    first = model.visual_tokens(pixels, context_length=1)
    assert first.shape[1] == 4
    assert model.last_vision_telemetry["offloaded"] is True
    second = model.visual_tokens(pixels, context_length=1)
    assert second.shape == first.shape
    assert model.last_vision_telemetry["cache_hit"] is True


def test_vision_output_adapter_selects_tipsv2_patch_tokens() -> None:
    cls = torch.randn(2, 1, 1024)
    registers = torch.randn(2, 1, 1024)
    patches = torch.randn(2, 1024, 1024)
    assert _extract_token_sequence((cls, registers, patches)) is patches

    from types import SimpleNamespace

    nested = SimpleNamespace(
        image_features=SimpleNamespace(
            cls_token=cls,
            register_tokens=registers,
            patch_tokens=patches,
        )
    )
    assert _extract_token_sequence(nested) is patches


def test_real_nemotron_wrapper_grafts_vision_projector_and_cache() -> None:
    config = KestrelConfig.tiny(use_vision=True)
    vision = InternViTEncoder(hidden_size=config.vision_hidden_size)
    projector = AdaptiveVisionProjector(
        config.vision_hidden_size, config.hidden_size, config.vision_token_budget
    )
    model = RealNemotronKestrelForCausalLM(
        config,
        torch.nn.Embedding(config.vocab_size, config.hidden_size),
        [],
        torch.nn.RMSNorm(config.hidden_size),
        torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        "test-nemotron",
        None,
        vision_encoder=vision,
        vision_projector=projector,
    )
    model.set_vision_trainable("projector")
    pixels = torch.rand(3, 20, 30)
    first = model.visual_tokens(pixels, budget=8, kind="ide")
    second = model.visual_tokens(pixels, budget=8, kind="ide")
    assert first.shape == (1, 8, config.hidden_size)
    assert torch.allclose(first, second)
    assert model.last_vision_telemetry["cache_hit"] is True
    assert all(parameter.requires_grad for parameter in projector.parameters())
    assert all(not parameter.requires_grad for parameter in vision.parameters())
    output = model(torch.ones(1, 4, dtype=torch.long), pixel_values=pixels, logits_to_keep=1)
    assert output.logits.shape == (1, 1, config.vocab_size)
