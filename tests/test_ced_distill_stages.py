"""Batch-2 tests for issue #9: five-stage schedule + resume-safe accounting.

CPU-only. Existing Batch-1 ``FreezePolicy`` stages and ``BudgetTracker``
behaviour is unchanged (see ``test_ced_distill.py``); this file covers only
the new five-stage API and ``TrainingRunState``.
"""

import pytest
import torch

from training.ced_distill import (
    BudgetTracker,
    FreezePolicy,
    STAGE_ORDER_5,
    STAGE_POLICIES_5,
    TrainingRunState,
    trainable_groups_5,
)


# --- five-stage schedule --------------------------------------------------


def test_stage_order_has_five_stages():
    assert STAGE_ORDER_5 == ("stage_1", "stage_2", "stage_3", "stage_4", "stage_5")


def test_stage_policies_have_complete_keys():
    required = {
        "encoder_frozen", "decoder_frozen", "bridge_trainable",
        "memory_gates_trainable", "global_kv_trainable", "indexer_trainable",
        "lora", "upper_encoder_layers_unfrozen", "selective_fullrank",
    }
    for stage in STAGE_ORDER_5:
        assert required <= set(STAGE_POLICIES_5[stage].keys()), stage


def test_stage_1_is_bridge_and_gates_only():
    policy = FreezePolicy.stage_policy_5("stage_1")
    assert policy["bridge_trainable"] and policy["memory_gates_trainable"]
    assert not policy["global_kv_trainable"]
    assert not policy["indexer_trainable"]
    assert not policy["lora"]
    assert policy["upper_encoder_layers_unfrozen"] == 0
    assert not policy["selective_fullrank"]
    assert trainable_groups_5(policy) == ["bridge", "memory_gates"]


def test_stage_2_adds_global_kv_and_indexer():
    policy = FreezePolicy.stage_policy_5("stage_2")
    assert policy["global_kv_trainable"] and policy["indexer_trainable"]
    assert not policy["lora"]
    assert trainable_groups_5(policy) == ["bridge", "global_kv", "indexer", "memory_gates"]


def test_stage_3_adds_decoder_lora():
    policy = FreezePolicy.stage_policy_5("stage_3")
    assert policy["lora"] and policy["decoder_frozen"]
    assert "lora" in trainable_groups_5(policy)


def test_stage_4_unfreezes_upper_encoder_only():
    policy = FreezePolicy.stage_policy_5("stage_4")
    assert policy["upper_encoder_layers_unfrozen"] == 2
    assert policy["decoder_frozen"]
    assert not policy["selective_fullrank"]
    assert "upper_encoder" in trainable_groups_5(policy)


def test_stage_5_is_narrow_selective_never_full_unfreeze():
    policy = FreezePolicy.stage_policy_5("stage_5")
    assert policy["selective_fullrank"]
    # Narrow: selective flag set, but no blanket full-model unfreeze marker.
    assert "full_unfreeze" not in policy
    assert "selective_fullrank" in trainable_groups_5(policy)


def test_trainable_groups_grow_monotonically():
    sizes = [len(trainable_groups_5(FreezePolicy.stage_policy_5(s))) for s in STAGE_ORDER_5]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_unknown_stage_raises_and_policy_is_a_copy():
    with pytest.raises(ValueError):
        FreezePolicy.stage_policy_5("stage_6")
    policy = FreezePolicy.stage_policy_5("stage_1")
    policy["lora"] = True
    assert not FreezePolicy.stage_policy_5("stage_1")["lora"]


def test_legacy_stages_untouched():
    assert FreezePolicy.stage_policy("bridge_only")["bridge_trainable"]
    with pytest.raises(ValueError):
        FreezePolicy.stage_policy("stage_1")


# --- TrainingRunState ------------------------------------------------------


def test_run_state_defaults():
    state = TrainingRunState()
    assert state.stage == "stage_1"
    assert state.tokens == 0
    assert state.gpu_hours == pytest.approx(0.0)
    assert state.checkpoint is None
    assert state.rng_state is None
    assert state.data_cursor == {}


def test_run_state_rejects_bad_init():
    with pytest.raises(ValueError):
        TrainingRunState(stage="stage_9")
    with pytest.raises(ValueError):
        TrainingRunState(tokens=-1)
    with pytest.raises(ValueError):
        TrainingRunState(gcp_cost_usd_est=-0.1)


def test_record_accumulates_all_meters():
    state = TrainingRunState()
    state.record(tokens=1000, wall_seconds=60.0, gpu_seconds=3600.0, gcp_cost_usd=1.5)
    state.record(tokens=500, wall_seconds=30.0, gpu_seconds=1800.0, gcp_cost_usd=0.75)
    assert state.tokens == 1500
    assert state.wall_seconds == pytest.approx(90.0)
    assert state.gpu_hours == pytest.approx(1.5)
    assert state.gcp_cost_usd_est == pytest.approx(2.25)
    with pytest.raises(ValueError):
        state.record(tokens=-1)
    with pytest.raises(ValueError):
        state.record(gpu_seconds=-1.0)


def test_advance_stage_is_strictly_forward():
    state = TrainingRunState()
    state.advance_stage("stage_2")
    assert state.stage == "stage_2"
    with pytest.raises(ValueError):
        state.advance_stage("stage_1")
    with pytest.raises(ValueError):
        state.advance_stage("stage_2")
    with pytest.raises(ValueError):
        state.advance_stage("stage_9")


def test_rng_capture_restore_roundtrip():
    state = TrainingRunState()
    torch.manual_seed(1234)
    state.capture_rng()
    assert state.rng_state and len(state.rng_state) > 0
    first = torch.randn(4)
    torch.randn(100)  # advance the RNG
    state.restore_rng()
    assert torch.equal(torch.randn(4), first)
    with pytest.raises(ValueError):
        TrainingRunState().restore_rng()


def test_checkpoint_and_data_cursor():
    state = TrainingRunState()
    state.set_checkpoint("checkpoints/stage1.pt")
    state.set_data_cursor({"shard": 3, "offset": 4096, "epoch": 0})
    assert state.checkpoint == "checkpoints/stage1.pt"
    assert state.data_cursor == {"shard": 3, "offset": 4096, "epoch": 0}


def test_save_load_roundtrip(tmp_path):
    state = TrainingRunState(stage="stage_3")
    state.record(tokens=2048, wall_seconds=120.0, gpu_seconds=600.0, gcp_cost_usd=0.4)
    state.set_checkpoint("ckpt/s3.pt")
    state.set_data_cursor({"shard": 1, "offset": 7})
    torch.manual_seed(99)
    state.capture_rng()
    expected = torch.randn(2)
    path = state.save(tmp_path / "run" / "state.json")
    loaded = TrainingRunState.load(path)
    assert loaded.to_dict() == state.to_dict()
    torch.randn(50)  # advance past the captured point
    loaded.restore_rng()
    assert torch.equal(torch.randn(2), expected)


def test_summary_reports_stage_groups():
    state = TrainingRunState(stage="stage_2")
    summary = state.summary()
    assert summary["stage"] == "stage_2"
    assert summary["trainable_groups"] == ["bridge", "global_kv", "indexer", "memory_gates"]
    assert "gpu_hours" in summary and "gcp_cost_usd_est" in summary


def test_budget_tracker_still_available():
    tracker = BudgetTracker(max_gpu_hours=50.0)
    tracker.record_stage("stage_1", gpu_hours=1.0, tokens=1000, peak_vram_gb=20.0)
    assert tracker.consumed_gpu_hours == pytest.approx(1.0)
