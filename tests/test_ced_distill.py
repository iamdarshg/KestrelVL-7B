"""Batch-1 tests for the CED mechanistic-distillation measurement skeleton.

CPU-only, tiny tensors, fast. Instrumentation only — no training runs.
"""

import json

import pytest
import torch

from training.ced_distill import (
    BudgetTracker,
    DistillLossConfig,
    FreezePolicy,
    bridge_memory_stats,
    compute_losses,
    measure_recovery,
    parameter_accounting,
    write_evidence_report,
)


def _logits(seed=0, rows=4, vocab=8, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=g) * scale


def _hiddens(layers, seed=0, rows=4, dim=8):
    out = {}
    for i in layers:
        g = torch.Generator().manual_seed(seed + i)
        out[i] = torch.randn(rows, dim, generator=g)
    return out


def _cfg(**overrides):
    kwargs = {"selected_layers": (0, 2)}
    kwargs.update(overrides)
    return DistillLossConfig(**kwargs)


# --- DistillLossConfig ---


def test_loss_config_defaults_validate():
    cfg = DistillLossConfig()
    assert cfg.temperature == 1.0
    assert len(cfg.selected_layers) > 0
    for w in (cfg.w_ce, cfg.w_kl, cfg.w_hidden, cfg.w_attn, cfg.w_kv, cfg.w_index):
        assert w >= 0.0
    cfg.validate()


def test_loss_config_negative_weight_raises():
    with pytest.raises(ValueError):
        DistillLossConfig(w_kl=-0.1).validate()


def test_loss_config_bad_temperature_raises():
    with pytest.raises(ValueError):
        DistillLossConfig(temperature=0.0).validate()


# --- compute_losses ---


def test_losses_zero_when_teacher_equals_student():
    t = _logits()
    h = _hiddens([0, 1, 2, 3])
    attn = _hiddens([0, 1, 2, 3])
    mem = {
        "teacher_kv": torch.ones(2, 4),
        "student_kv": torch.ones(2, 4),
        "teacher_index": torch.ones(2, 4),
        "student_index": torch.ones(2, 4),
    }
    cfg = _cfg()
    losses, log = compute_losses(t, t.clone(), h, {k: v.clone() for k, v in h.items()},
                                 attn, {k: v.clone() for k, v in attn.items()},
                                 mem, None, cfg)
    for key in ("ce", "kl", "hidden", "attn", "kv", "index", "total"):
        assert key in losses
        assert losses[key] == pytest.approx(0.0, abs=1e-6)
    # per-term logging present with active flags
    for key in ("ce", "kl", "hidden", "attn", "kv", "index"):
        assert f"{key}_active" in log


def test_kl_positive_when_logits_differ():
    t = _logits(seed=0)
    s = _logits(seed=1)
    cfg = _cfg()
    losses, log = compute_losses(t, s, {}, {}, None, None, None, None, cfg)
    assert losses["kl"] > 0.0
    assert log["kl_active"] is True
    assert log["ce_active"] is False  # no labels -> inactive, zero
    assert losses["ce"] == 0.0


def test_ce_active_with_labels():
    t = _logits()
    s = _logits(seed=1)
    labels = torch.tensor([1, 2, 3, 0])
    cfg = _cfg()
    losses, log = compute_losses(t, s, {}, {}, None, None, None, labels, cfg)
    assert losses["ce"] > 0.0
    assert log["ce_active"] is True


def test_weights_respected_in_total():
    t = _logits(seed=0)
    s = _logits(seed=1)
    base = _cfg(w_ce=0.0, w_kl=1.0, w_hidden=0.0, w_attn=0.0, w_kv=0.0, w_index=0.0)
    doubled = _cfg(w_ce=0.0, w_kl=2.0, w_hidden=0.0, w_attn=0.0, w_kv=0.0, w_index=0.0)
    l1, _ = compute_losses(t, s, {}, {}, None, None, None, None, base)
    l2, _ = compute_losses(t, s, {}, {}, None, None, None, None, doubled)
    assert l2["total"] == pytest.approx(2.0 * l1["total"], rel=1e-5)


def test_selected_layer_subset_respected():
    layers = [0, 1, 2, 3]
    th = _hiddens(layers, seed=0)
    sh = {k: v.clone() for k, v in th.items()}
    # perturb only UNSELECTED layers -> hidden term must stay zero
    sh[1] = sh[1] + 100.0
    sh[3] = sh[3] - 100.0
    cfg = _cfg()
    losses, _ = compute_losses(_logits(), _logits(), th, sh, None, None, None, None, cfg)
    assert losses["hidden"] == pytest.approx(0.0, abs=1e-6)
    # perturb a SELECTED layer -> hidden term becomes positive
    sh[0] = sh[0] + 5.0
    losses2, _ = compute_losses(_logits(), _logits(), th, sh, None, None, None, None,
                                cfg)
    assert losses2["hidden"] > 0.0


def test_missing_optional_inputs_give_zero_terms():
    t = _logits(seed=0)
    s = _logits(seed=1)
    cfg = _cfg()
    losses, log = compute_losses(t, s, {}, {}, None, None, None, None, cfg)
    assert losses["hidden"] == 0.0 and log["hidden_active"] is False
    assert losses["attn"] == 0.0 and log["attn_active"] is False
    assert losses["kv"] == 0.0 and log["kv_active"] is False
    assert losses["index"] == 0.0 and log["index_active"] is False


# --- measure_recovery ---


def test_recovery_identical_within_tol():
    t = _logits()
    out = measure_recovery(t, t.clone())
    assert out["within_tol"] is True
    assert out["max_abs_diff"] == pytest.approx(0.0, abs=1e-9)
    assert out["mean_abs_diff"] == pytest.approx(0.0, abs=1e-9)


def test_recovery_perturbed_outside_tol_quantified():
    t = _logits()
    shift = torch.linspace(0.0, 1.0, steps=t.numel()).reshape_as(t)
    out = measure_recovery(t, t + shift)
    assert out["within_tol"] is False
    assert out["max_abs_diff"] == pytest.approx(1.0, rel=1e-5)
    assert out["mean_abs_diff"] == pytest.approx(0.5, rel=1e-5)
    assert out["kl"] > 0.0


# --- bridge_memory_stats ---


def test_memory_stats_deterministic_fingerprint():
    k = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    v = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 0.5
    a = bridge_memory_stats(k, v)
    b = bridge_memory_stats(k.clone(), v.clone())
    assert a["fingerprint"] == b["fingerprint"]
    for key in ("norm_k", "norm_v", "mean_k", "std_k", "mean_v", "std_v",
                "rank_proxy", "fingerprint"):
        assert key in a
    c = bridge_memory_stats(k + 1.0, v)
    assert c["fingerprint"] != a["fingerprint"]


# --- parameter_accounting ---


def test_parameter_accounting_sums():
    params = [
        ("encoder.layer0.weight", True, 100),
        ("encoder.layer0.bias", False, 10),
        ("decoder.head.weight", True, 50),
    ]
    out = parameter_accounting(params)
    assert out["trainable"] == 150
    assert out["frozen"] == 10
    assert out["total"] == 160
    assert "encoder" in out["by_prefix"] and "decoder" in out["by_prefix"]


# --- FreezePolicy ---


def test_freeze_bridge_only_freezes_encoder_decoder():
    pol = FreezePolicy.stage_policy("bridge_only")
    assert pol["encoder"] is True
    assert pol["decoder"] is True
    assert pol["bridge_trainable"] is True
    assert pol["lora"] is False


def test_freeze_unknown_stage_raises():
    with pytest.raises(ValueError):
        FreezePolicy.stage_policy("nope")


def test_freeze_all_stages_have_keys():
    for stage in ("bridge_only", "bridge_global_kv", "lora_recovery", "selective_unfreeze"):
        pol = FreezePolicy.stage_policy(stage)
        assert set(pol) == {"encoder", "decoder", "bridge_trainable", "lora"}


# --- BudgetTracker ---


def test_budget_accumulates_and_summary(tmp_path):
    bt = BudgetTracker()
    bt.record_stage("s1", gpu_hours=2.0, tokens=1000, peak_vram_gb=10.0)
    bt.record_stage("s2", gpu_hours=3.0, tokens=2000, peak_vram_gb=12.0)
    assert bt.consumed_gpu_hours == pytest.approx(5.0)
    s = bt.summary()
    assert s["consumed_gpu_hours"] == pytest.approx(5.0)
    assert s["consumed_tokens"] == 3000
    assert len(s["stages"]) == 2


def test_budget_hard_stop_raises():
    bt = BudgetTracker(max_gpu_hours=1.0, hard_stop=True)
    with pytest.raises(RuntimeError):
        bt.record_stage("big", gpu_hours=2.0, tokens=10, peak_vram_gb=1.0)


def test_budget_save_load_preserves_consumed(tmp_path):
    bt = BudgetTracker(max_gpu_hours=50.0, gpu_class="a100_80gb", hard_stop=True)
    bt.record_stage("s1", gpu_hours=4.0, tokens=500, peak_vram_gb=8.0)
    path = tmp_path / "budget.json"
    bt.save(path)
    bt2 = BudgetTracker.load(path)
    assert bt2.consumed_gpu_hours == pytest.approx(4.0)
    assert bt2.consumed_tokens == 500
    assert bt2.max_gpu_hours == 50.0
    # loaded tracker still enforces the hard stop
    with pytest.raises(RuntimeError):
        bt2.record_stage("over", gpu_hours=100.0, tokens=1, peak_vram_gb=1.0)


# --- write_evidence_report ---


def test_write_evidence_report_deterministic(tmp_path):
    p1 = tmp_path / "ev1.json"
    p2 = tmp_path / "ev2.json"
    payload = {"b": [1, 2], "a": 1}
    r1 = write_evidence_report(p1, payload)
    r2 = write_evidence_report(p2, payload)
    assert r1["sha256"] == r2["sha256"]
    raw = json.loads(p1.read_text(encoding="utf-8"))
    assert list(raw) == sorted(raw)  # sorted keys on disk
