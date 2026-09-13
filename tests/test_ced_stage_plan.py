"""Batch-3 CPU gate: stage YAML matches the harness schedule + budget envelope.

No GPU, no weights. Fails closed if `configs/model/qwen35_ced.yaml` drifts
from `STAGE_POLICIES_5` / `STAGE_ORDER_5` / `BudgetTracker` semantics.
"""

import os

import pytest
import torch
import yaml

from training.ced_distill import (
    BudgetTracker,
    DistillLossConfig,
    FreezePolicy,
    STAGE_ORDER_5,
    TrainingRunState,
    trainable_groups_5,
)

YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "model", "qwen35_ced.yaml"
)


def _config():
    with open(YAML_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_yaml_stages_match_harness_order():
    cfg = _config()
    assert list(cfg["training_stages"].keys()) == list(STAGE_ORDER_5)


def test_yaml_hour_splits_fit_budget():
    cfg = _config()
    max_hours = float(cfg["compute_budget"]["max_gpu_hours"])
    total = sum(float(v["gpu_hours"]) for v in cfg["training_stages"].values())
    assert total <= max_hours
    assert all(float(v["gpu_hours"]) >= 0 for v in cfg["training_stages"].values())


def test_yaml_loss_matches_distill_defaults():
    cfg = _config()
    loss = DistillLossConfig(**{k: v for k, v in cfg["loss"].items()
                                if k != "selected_layers"},
                             selected_layers=tuple(cfg["loss"]["selected_layers"]))
    assert loss.w_ce == pytest.approx(1.0)
    assert loss.temperature == pytest.approx(1.0)


def test_full_walk_all_five_stages_with_resume():
    """End-to-end stage walk on CPU: advance, record meters, checkpoint, resume."""
    tracker = BudgetTracker(max_gpu_hours=50.0, gpu_class="a100_80gb", hard_stop=True)
    cfg = _config()
    state = TrainingRunState()
    torch.manual_seed(2026)
    state.capture_rng()
    for stage in STAGE_ORDER_5:
        assert state.stage == stage
        policy = FreezePolicy.stage_policy_5(stage)
        assert trainable_groups_5(policy)  # every stage trains SOMETHING
        hours = float(cfg["training_stages"][stage]["gpu_hours"])
        state.record(tokens=1000, wall_seconds=10.0, gpu_seconds=hours * 3600.0,
                     gcp_cost_usd=hours * 5.03)
        tracker.record_stage(stage, gpu_hours=hours, tokens=1000, peak_vram_gb=70.0)
        state.set_checkpoint(f"checkpoints/{stage}.pt")
        state.set_data_cursor({"shard": 0, "offset": 1000})
        if stage != STAGE_ORDER_5[-1]:
            nxt = STAGE_ORDER_5[STAGE_ORDER_5.index(stage) + 1]
            state.advance_stage(nxt)
    assert state.stage == "stage_5"
    assert state.gpu_hours == pytest.approx(50.0)
    assert tracker.consumed_gpu_hours == pytest.approx(50.0)
    # Resume from the saved dict: identical accounting, RNG restorable.
    resumed = TrainingRunState.from_dict(state.to_dict())
    assert resumed.to_dict() == state.to_dict()
    resumed.restore_rng()  # must not raise
    # Budget hard-stop still enforced past the envelope.
    with pytest.raises(RuntimeError):
        tracker.record_stage("overrun", gpu_hours=0.1, tokens=1, peak_vram_gb=1.0)


def test_stage5_yaml_marks_opt_in():
    cfg = _config()
    assert "opt-in" in cfg["training_stages"]["stage_5"]["focus"]
