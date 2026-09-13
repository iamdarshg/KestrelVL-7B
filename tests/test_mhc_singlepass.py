"""Batch-2 tests for issue #7: Single-Pass mHC selectable backend.

CPU-only, tiny tensors. The old sequential mHC pair is retained for A/B;
Single-Pass is opt-in and evidence-gated (never default).
"""

import pytest
import torch
from torch import nn

from model.attention.mhc import ManifoldHyperConnection
from model.attention.mhc_singlepass import (
    MHC_BACKENDS,
    SinglePassMHC,
    build_single_pass_from_pair,
    mixed_attn_state,
    resolve_mhc_backend,
)
from model.ced import ExternalMemoryHook
from model.configuration import KestrelConfig
from model.nemotron import RealDecoderLayer


def _tensors(seed=0, b=2, t=4, d=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(b, t, d, generator=g)
    g = torch.Generator().manual_seed(seed + 1)
    attn = torch.randn(b, t, d, generator=g) * 0.5
    g = torch.Generator().manual_seed(seed + 2)
    mlp = torch.randn(b, t, d, generator=g) * 0.5
    return base, attn, mlp


def _pair(seed=0, streams=2):
    torch.manual_seed(seed)
    a = ManifoldHyperConnection(streams=streams, sinkhorn_iters=6, enabled=True)
    torch.manual_seed(seed + 100)
    m = ManifoldHyperConnection(streams=streams, sinkhorn_iters=6, enabled=True)
    return a, m


def _sequential(a, m, base, attn, mlp):
    x1 = a(base, attn)
    return m(x1, mlp)


# --- backend selection ----------------------------------------------------


def test_backend_default_is_residual():
    assert KestrelConfig.tiny().mhc_backend == "residual"
    assert MHC_BACKENDS[0] == "residual"


def test_backend_invalid_raises():
    with pytest.raises(ValueError):
        KestrelConfig.tiny(mhc_backend="turbo")
    with pytest.raises(ValueError):
        resolve_mhc_backend("turbo")
    assert resolve_mhc_backend("single_pass") == "single_pass"


# --- identity / reference initialisation ----------------------------------


def test_fresh_fused_matches_fresh_sequential_pair():
    base, attn, mlp = _tensors()
    a, m = _pair(seed=7)
    fused = SinglePassMHC.from_sequential(a, m)
    expected = _sequential(a, m, base, attn, mlp)
    got = fused(base, attn, mlp)
    assert torch.allclose(got, expected, atol=1e-6)


def test_from_sequential_copies_trained_values():
    base, attn, mlp = _tensors()
    a, m = _pair(seed=3)
    with torch.no_grad():
        a.logits += 0.25
        m.residual_scale += 0.5
    fused = build_single_pass_from_pair(a, m)
    assert torch.allclose(fused(base, attn, mlp), _sequential(a, m, base, attn, mlp), atol=1e-6)


def test_disabled_is_plain_residual_sum():
    base, attn, mlp = _tensors()
    fused = SinglePassMHC(enabled=False)
    assert torch.allclose(fused(base, attn, mlp), base + attn + mlp, atol=1e-7)
    a, m = _pair()
    a.enabled = False
    m.enabled = False
    assert torch.allclose(_sequential(a, m, base, attn, mlp), base + attn + mlp, atol=1e-7)


def test_mismatched_shapes_raise():
    fused = SinglePassMHC()
    b, a, m = _tensors()
    with pytest.raises(ValueError):
        fused(b, a, m[..., :4])


def test_stream_mismatch_raises():
    a, m = _pair()
    m3 = ManifoldHyperConnection(streams=3)
    with pytest.raises(ValueError):
        SinglePassMHC.from_sequential(a, m3)


# --- hidden-state delta measurement ---------------------------------------


def test_hidden_state_delta_is_zero_at_matched_init():
    base, attn, mlp = _tensors(seed=11)
    a, m = _pair(seed=11)
    fused = SinglePassMHC.from_sequential(a, m)
    delta = (fused(base, attn, mlp) - _sequential(a, m, base, attn, mlp)).detach().abs()
    assert float(delta.max()) == pytest.approx(0.0, abs=1e-6)
    assert float(delta.mean()) == pytest.approx(0.0, abs=1e-7)


def test_delta_grows_when_logits_drift():
    base, attn, mlp = _tensors(seed=5)
    a, m = _pair(seed=5)
    fused = SinglePassMHC.from_sequential(a, m)
    with torch.no_grad():
        # Non-uniform perturbation: Sinkhorn is invariant to uniform shifts,
        # so the drift must break row/column symmetry to move the matrix.
        fused.logits_mlp[0, 0] += 2.0
        fused.logits_mlp[1, 0] -= 2.0
    delta = float((fused(base, attn, mlp) - _sequential(a, m, base, attn, mlp)).detach().abs().max())
    assert delta > 1e-4


# --- gradient flow / freezing ---------------------------------------------


def test_gradient_flows_to_all_fused_params():
    base, attn, mlp = _tensors()
    for t in (base, attn, mlp):
        t.requires_grad_(True)
    fused = SinglePassMHC()
    loss = fused(base, attn, mlp).square().mean()
    loss.backward()
    names = {n for n, p in fused.named_parameters()}
    assert names == {"logits_attn", "logits_mlp", "residual_scale_attn", "residual_scale_mlp"}
    for p in fused.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_freeze_unfreeze_roundtrip():
    fused = SinglePassMHC()
    fused.freeze()
    assert all(not p.requires_grad for p in fused.parameters())
    base, attn, mlp = _tensors()
    out = fused(base, attn, mlp)  # forward still works frozen
    assert out.shape == base.shape
    fused.unfreeze()
    assert all(p.requires_grad for p in fused.parameters())


# --- checkpoint / resume ---------------------------------------------------


def test_snapshot_roundtrip_and_metadata_guard():
    fused = SinglePassMHC()
    snap = fused.state_dict_snapshot()
    meta = fused.metadata()
    assert meta["backend"] == "single_pass"
    other = SinglePassMHC()
    with torch.no_grad():
        other.logits_attn.zero_()
    other.load_snapshot_strict(dict(snap), dict(meta))
    base, attn, mlp = _tensors()
    assert torch.allclose(other(base, attn, mlp), fused(base, attn, mlp), atol=1e-7)
    bad = dict(meta)
    bad["streams"] = 99
    with pytest.raises(ValueError):
        other.load_snapshot_strict(dict(snap), bad)


# --- eager / coefficient agreement ----------------------------------------


def test_coefficients_match_manual_sinkhorn_math():
    torch.manual_seed(2)
    fused = SinglePassMHC()
    coef = {k: v.detach() for k, v in fused.coefficients().items()}
    ma = fused.matrices()[0].detach().to(torch.float32)
    mm = fused.matrices()[1].detach().to(torch.float32)
    sa = float(fused.residual_scale_attn.detach())
    sm = float(fused.residual_scale_mlp.detach())
    assert float(coef["alpha"]) == pytest.approx((1 + sm * mm[0, 0]) * (1 + sa * ma[0, 0]), rel=1e-5)
    assert float(coef["beta"]) == pytest.approx((1 + sm * mm[0, 0]) * (sa * ma[0, 1]), rel=1e-5)
    assert float(coef["gamma"]) == pytest.approx(sm * mm[0, 1], rel=1e-5)


def test_param_count_equals_replaced_pair():
    fused = SinglePassMHC(streams=2)
    a, m = _pair()
    n_fused = sum(p.numel() for p in fused.parameters())
    n_pair = sum(p.numel() for p in list(a.parameters()) + list(m.parameters()))
    assert n_fused == n_pair


def test_no_unbounded_per_token_state_and_prefix_stable():
    fused = SinglePassMHC()
    assert list(fused.buffers()) == []
    for name, value in vars(fused).items():
        assert not isinstance(value, torch.Tensor), name
    base, attn, mlp = _tensors(seed=9, t=8)
    full = fused(base, attn, mlp)
    first = fused(base[:, :1], attn[:, :1], mlp[:, :1])
    assert torch.allclose(full[:, :1], first, atol=1e-7)


# --- CED external-memory interaction --------------------------------------


def test_ced_hook_plus_single_pass_forward_backward():
    torch.manual_seed(0)
    hook = ExternalMemoryHook(decoder_dim=16, encoder_dim=16)
    fused = SinglePassMHC()
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    memory = torch.randn(2, 7, 16)
    # Zero-gate baseline stays intact: a fresh hook (gate exactly 0.0) is
    # bit-exact identity.
    assert torch.equal(hook(hidden.detach(), memory), hidden.detach())
    with torch.no_grad():
        hook.gate.fill_(0.1)
    hooked = hook(hidden, memory)
    attn_update = torch.randn(2, 5, 16) * 0.1
    out = fused(hooked, attn_update, torch.zeros_like(hooked))
    loss = out.square().mean()
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(out).all()
    assert hook.gate.grad is not None


# --- RealDecoderLayer routing ----------------------------------------------


class _StubAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, position_ids=None, cache=None):
        return self.proj(x), torch.zeros_like(x)


def _decoder_layer(backend="residual", seed=0):
    torch.manual_seed(seed)
    dim = 64
    return RealDecoderLayer(
        input_norm=nn.LayerNorm(dim),
        post_norm=nn.LayerNorm(dim),
        mlp=nn.Linear(dim, dim),
        attention=_StubAttn(dim),
        config=KestrelConfig.tiny(mhc_backend=backend),
    )


def test_decoder_layer_defaults_to_residual_pair():
    layer = _decoder_layer("residual")
    assert layer.mhc_backend == "residual"
    assert layer.mhc_fused is None
    x = torch.randn(1, 4, 64)
    out, branch = layer(x, torch.arange(4).view(1, -1), None)
    assert out.shape == x.shape and branch.shape == x.shape


def test_decoder_layer_single_pass_agrees_with_residual():
    torch.manual_seed(4)
    residual = _decoder_layer("residual", seed=4)
    sp = _decoder_layer("single_pass", seed=4)
    assert sp.mhc_fused is not None
    # Align weights: copy the residual pair's live params into the fused module.
    sp.attn_mhc.load_state_dict(residual.attn_mhc.state_dict())
    sp.mlp_mhc.load_state_dict(residual.mlp_mhc.state_dict())
    sp.mhc_fused = SinglePassMHC.from_sequential(sp.attn_mhc, sp.mlp_mhc)
    for p, q in zip(sp.parameters(), residual.parameters()):
        pass  # layer weights differ by seed path; only mHC pair was aligned
    x = torch.randn(1, 4, 64)
    # Copy non-mHC weights so the ONLY difference is the mixing backend.
    sp.input_norm.load_state_dict(residual.input_norm.state_dict())
    sp.post_norm.load_state_dict(residual.post_norm.state_dict())
    sp.mlp.load_state_dict(residual.mlp.state_dict())
    sp.attention.load_state_dict(residual.attention.state_dict())
    pos = torch.arange(4).view(1, -1)
    out_r, _ = residual(x, pos, None)
    out_s, _ = sp(x, pos, None)
    assert torch.allclose(out_s, out_r, atol=1e-5)


def test_promote_to_single_pass_is_explicit():
    layer = _decoder_layer("residual", seed=6)
    assert layer.mhc_backend == "residual"
    fused = layer.promote_to_single_pass()
    assert isinstance(fused, SinglePassMHC)
    # Promotion alone does NOT reroute the forward path.
    assert layer.mhc_backend == "residual"
    x = torch.randn(1, 3, 64)
    out, _ = layer(x, torch.arange(3).view(1, -1), None)
    assert out.shape == x.shape


def test_mixed_attn_state_matches_pair_first_stage():
    base, attn, _ = _tensors()
    a, _ = _pair(seed=1)
    assert torch.allclose(mixed_attn_state(base, attn, a), a(base, attn), atol=1e-7)
