import pytest
import torch

from model import KestrelConfig, KestrelForCausalLM
from model.attention.cache import KestrelCache
from model.attention.csa import CompressedSparseAttention
from model.attention.lightning_indexer import LightningIndexer
from model.attention.mhc import ManifoldHyperConnection
from model.attention.rope import PartialRotaryEmbedding
from model.attention.sliding import sliding_causal_attention


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
