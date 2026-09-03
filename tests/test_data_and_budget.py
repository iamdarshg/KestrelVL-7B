from pathlib import Path
import dataclasses
import json

import pytest
import torch

from data.corpus import CompositionLockedCorpus, CorpusSpec
from data.contamination import ContaminationIndex, normalize_code
from data.fim import make_fim, should_fim
from data.manifests import ManifestStore, SampleRecord
from data.real_corpus import RealCorpusSpec, RealSourceSpec, RealStreamingCorpus
from eval.recovery import evaluate_teacher_recovery
from model import KestrelConfig, KestrelForCausalLM
from training.checkpoint import SafeCheckpointManager
from training.budget import check_budget, conservative_cost
from training.muon import Muon


def test_manifest_round_trip_and_digest(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "provenance.sqlite")
    store.add(SampleRecord("a", "synthetic", "a", "AGPL-3.0", "text", 0.9))
    assert list(store)[0].sample_id == "a"
    assert len(store.digest()) == 64
    store.close()


def test_contamination_rejects_exact_and_holdout() -> None:
    index = ContaminationIndex()
    index.add("known", "def f():\n  return 1")
    assert index.check("new", "def f(): return 1") == "exact_duplicate"
    index.add_held_out("gpqa-1")
    assert index.check("gpqa-1", "anything") == "benchmark_holdout"
    assert normalize_code("a\n b") == "a b"


def test_fim_is_reproducible_and_bounded() -> None:
    first = make_fim("a", "b", "c")
    assert first.startswith("<|fim_prefix|>")
    assert any(should_fim(i) for i in range(10))


def test_muon_updates_matrices_and_vectors_finitely() -> None:
    matrix = torch.nn.Parameter(torch.eye(4))
    vector = torch.nn.Parameter(torch.ones(4))
    optimizer = Muon([{"params": [matrix]}, {"params": [vector]}], lr=0.02, adamw_lr=1e-4)
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    assert torch.isfinite(matrix).all() and torch.isfinite(vector).all()
    assert not torch.equal(matrix, torch.eye(4))


def test_muon_handles_grouped_matrix_parameters() -> None:
    grouped = torch.nn.Parameter(torch.eye(4).repeat(2, 1, 1))
    optimizer = Muon([grouped], lr=0.02, adamw_lr=1e-4)
    grouped.grad = torch.ones_like(grouped)
    optimizer.step()
    assert torch.isfinite(grouped).all()
    assert grouped.shape == (2, 4, 4)


def test_muon_routes_oversized_matrices_to_adamw_fallback() -> None:
    oversized = torch.nn.Parameter(torch.ones(5000, 2))
    model = torch.nn.Module()
    model.register_parameter("oversized", oversized)
    optimizer = Muon(
        [{"params": [oversized], "name": "fallback", "use_muon": False}],
        lr=0.02,
        adamw_lr=1e-4,
    )
    oversized.grad = torch.ones_like(oversized)
    optimizer.step()
    assert torch.isfinite(oversized).all()


def test_corpus_composition_is_reproducible() -> None:
    spec = CorpusSpec(total_ablation_token_budget=10_000, validation_token_budget=100, source_block_tokens=1)
    corpus_a = CompositionLockedCorpus(spec, vocab_size=257)
    corpus_b = CompositionLockedCorpus(spec, vocab_size=257)
    assert corpus_a.source_histogram(20) == {
        "stack-edu": 7,
        "refinecode": 5,
        "stack-v2": 4,
        "docs": 2,
        "history": 2,
    }
    assert torch.equal(corpus_a.batch(0, 32)[0], corpus_b.batch(0, 32)[0])
    assert not torch.equal(corpus_a.batch(0, 32)[0], corpus_a.batch(0, 32, validation=True)[0])


def test_safe_checkpoint_round_trip_has_no_pickle_files(tmp_path: Path) -> None:
    torch.manual_seed(18)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    manager = SafeCheckpointManager(tmp_path / "resume", interval_steps=1, max_checkpoints=1)
    checkpoint = manager.save(
        7,
        model,
        optimizer,
        dataset_state={"cursor": 123},
        metrics={"tokens": 456},
    )
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "optimizer.safetensors").exists()
    assert (checkpoint / "manifest.json").exists()
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "checksums.json").exists()
    assert not list(checkpoint.glob("*.pt"))

    restored_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    state = manager.load(checkpoint, restored_model, restored_optimizer)
    assert state["step"] == 7
    for left, right in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(left, right)

    model_bytes = (checkpoint / "model.safetensors").read_bytes()
    try:
        (checkpoint / "model.safetensors").write_bytes(model_bytes + b"corruption")
        with pytest.raises(ValueError, match="checksum mismatch"):
            manager.load(checkpoint, restored_model)
    finally:
        (checkpoint / "model.safetensors").write_bytes(model_bytes)


def _real_fixture_spec(tmp_path: Path, revision: str = "fixture-v1") -> RealCorpusSpec:
    sources = []
    for index, name in enumerate(("stack-edu", "refinecode", "stack-v2", "docs", "history")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps({
                "id": f"{name}-0", "repo_name": f"repo-{name}",
                "license": "MIT", "text": f"def {name.replace('-', '_')}_zero(): return {index}\n"
            }) + "\n" + json.dumps({
                "id": f"{name}-1", "repo_name": f"repo-{name}-one",
                "license": "MIT", "text": f"def {name.replace('-', '_')}_one(): return {index + 1}\n"
            }) + "\n",
            encoding="utf-8",
        )
        sources.append(RealSourceSpec(name, (0.35, 0.25, 0.20, 0.10, 0.10)[index], "jsonl", str(path), revision))
    return RealCorpusSpec(
        "fixture", 7, 32, 1, 0.35, "fixture-tokenizer", tuple(sources),
        architecture_validation_fraction=0.0, recovery_validation_fraction=0.0, min_chars=1,
    )


def test_real_corpus_fingerprint_and_resume_are_deterministic(tmp_path: Path) -> None:
    spec = _real_fixture_spec(tmp_path)
    changed = _real_fixture_spec(tmp_path, revision="fixture-v2")
    assert spec.fingerprint() != changed.fingerprint()
    first = RealStreamingCorpus(spec)
    record_a, text_a = first.next_record(0)
    state = first.state_dict()
    second = RealStreamingCorpus(spec)
    second.load_state_dict(state)
    assert record_a.source == "stack-edu"
    assert text_a.startswith("def stack_edu_zero")
    assert second.state_dict()["manifest_fingerprint"] == spec.fingerprint()


def test_real_corpus_fails_closed_for_missing_source(tmp_path: Path) -> None:
    spec = _real_fixture_spec(tmp_path)
    missing = RealSourceSpec("stack-edu", 0.35, "jsonl", str(tmp_path / "missing.jsonl"), "v1")
    spec = dataclasses.replace(spec, source_specs=(missing,) + spec.source_specs[1:])
    with pytest.raises(RuntimeError, match="missing"):
        RealStreamingCorpus(spec).next_record(0)


def test_recovery_metrics_are_zero_for_identical_tiny_models() -> None:
    config = KestrelConfig.tiny(use_vision=False)
    teacher = KestrelForCausalLM(config).eval()
    candidate = KestrelForCausalLM(config).eval()
    candidate.load_state_dict(teacher.state_dict())
    ids = torch.randint(0, config.vocab_size, (1, 16))
    report = evaluate_teacher_recovery(teacher, candidate, ids)
    assert report["finite"] is True
    assert abs(float(report["delta_nll"])) < 1e-6
    assert float(report["forward_kl_teacher_to_candidate"]) < 1e-6
    assert float(report["hidden_state"]["0"]["cosine"]) > 0.99999
    assert report["teacher_capability_retention"] is None


def test_budget_guard_counts_conservative_projection(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"entries": [{"estimated_cost_usd": 29.0}]}), encoding="utf-8")
    decision = check_budget(ledger, conservative_cost(1.0, 1.0), 30.0)
    assert decision.allowed is False
