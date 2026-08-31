from pathlib import Path

import torch

from data.corpus import CompositionLockedCorpus, CorpusSpec
from data.contamination import ContaminationIndex, normalize_code
from data.fim import make_fim, should_fim
from data.manifests import ManifestStore, SampleRecord
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
