"""Deterministic composition-locked corpus streams for fair ablations.

The local fallback is a synthetic token stream with the same source-mixture
schedule as the planned final CPT corpus.  It exists to validate optimizer,
checkpoint, comparison, and continuation mechanics without pretending that
synthetic tokens are equivalent to Stack-Edu/RefineCode/The Stack data.
Replace it with a governed JSONL/WebDataset reader for production runs while
keeping the composition file and fingerprint unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


FINAL_SOURCE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("stack-edu", 0.35),
    ("refinecode", 0.25),
    ("stack-v2", 0.20),
    ("docs", 0.10),
    ("history", 0.10),
)


@dataclass(frozen=True)
class CorpusSpec:
    name: str = "kestrel-final-cpt-v1"
    total_ablation_token_budget: int = 10_000_000
    validation_token_budget: int = 100_000
    seed: int = 20260831
    sequence_length: int = 1024
    source_block_tokens: int = 256
    fim_rate: float = 0.35
    source_weights: tuple[tuple[str, float], ...] = FINAL_SOURCE_WEIGHTS

    def __post_init__(self) -> None:
        if self.total_ablation_token_budget <= 0 or self.validation_token_budget <= 0:
            raise ValueError("corpus token budgets must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.source_block_tokens < 1:
            raise ValueError("source_block_tokens must be positive")
        if not 0.0 <= self.fim_rate <= 1.0:
            raise ValueError("fim_rate must be in [0, 1]")
        total = sum(weight for _, weight in self.source_weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to one, got {total}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CorpusSpec":
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        sources = tuple((str(name), float(weight)) for name, weight in payload["sources"].items())
        transforms = payload.get("transforms", {})
        return cls(
            name=str(payload.get("name", cls.name)),
            total_ablation_token_budget=int(payload.get("total_ablation_token_budget", cls.total_ablation_token_budget)),
            validation_token_budget=int(payload.get("validation_token_budget", cls.validation_token_budget)),
            seed=int(payload.get("seed", cls.seed)),
            sequence_length=int(payload.get("sequence_length", cls.sequence_length)),
            source_block_tokens=int(payload.get("source_block_tokens", cls.source_block_tokens)),
            fim_rate=float(transforms.get("fim_rate", cls.fim_rate)),
            source_weights=sources,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_ablation_token_budget": self.total_ablation_token_budget,
            "validation_token_budget": self.validation_token_budget,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "source_block_tokens": self.source_block_tokens,
            "fim_rate": self.fim_rate,
            "sources": dict(self.source_weights),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class CompositionLockedCorpus:
    """A reproducible token stream whose source proportions are exact per cycle."""

    def __init__(self, spec: CorpusSpec, vocab_size: int, tokenizer: object | None = None) -> None:
        self.spec = spec
        self.vocab_size = vocab_size
        if vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        self._tokenizer = tokenizer
        # Convert rational-looking weights to the shortest exact cycle.  The
        # configured 35/25/20/10/10 mixture becomes a 20-example cycle.
        denominator = 100
        counts: list[tuple[str, int]] = []
        for source, weight in spec.source_weights:
            count = round(weight * denominator)
            counts.append((source, count))
        divisor = 0
        for _, count in counts:
            divisor = math.gcd(divisor, count)
        slots: list[str] = []
        for source, count in counts:
            slots.extend([source] * (count // max(divisor, 1)))
        if not slots:
            raise ValueError("source mixture has no slots")
        self._cycle = tuple(slots)
        self._source_ids = {source: index + 1 for index, (source, _) in enumerate(spec.source_weights)}
        self._source_tokens = self._build_source_tokens(tokenizer)

    def _build_source_tokens(self, tokenizer: object | None) -> dict[str, torch.Tensor] | None:
        if tokenizer is None:
            return None
        templates = {
            "stack-edu": "# educational implementation\ndef solve(problem):\n    \"\"\"Explain the invariant, handle edge cases, and return a result.\"\"\"\n    return problem\n",
            "refinecode": "def parse_config(text):\n    lines = [line.strip() for line in text.splitlines() if line.strip()]\n    return {key: value for key, value in (line.split('=', 1) for line in lines)}\n",
            "stack-v2": "class RepositoryService:\n    def __init__(self, client):\n        self.client = client\n\n    def fetch(self, identifier):\n        return self.client.get(identifier)\n",
            "docs": "## API contract\nCall the parser before validation. Return a structured error with the file, line, and remediation.\n",
            "history": "diff --git a/module.py b/module.py\n@@\n-    return old_value\n+    return new_value\n# tests: pytest -q\n",
        }
        result: dict[str, torch.Tensor] = {}
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise TypeError("tokenizer must expose encode(text, add_special_tokens=...)")
        for source, _ in self.spec.source_weights:
            ids = encode(templates.get(source, templates["docs"]), add_special_tokens=False)
            if not ids:
                raise ValueError(f"tokenizer produced no tokens for {source}")
            result[source] = torch.tensor(ids, dtype=torch.long)
        return result

    def source_at(self, token_index: int, validation: bool = False) -> str:
        # A validation phase uses a disjoint cycle rotation and seed domain but
        # preserves exactly the same source counts.
        block = token_index // self.spec.source_block_tokens
        offset = (self.spec.seed + (7919 if validation else 0)) % len(self._cycle)
        return self._cycle[(block + offset) % len(self._cycle)]

    def source_histogram(self, token_count: int, validation: bool = False) -> dict[str, int]:
        counts = {source: 0 for source, _ in self.spec.source_weights}
        for index in range(token_count):
            source = self.source_at(index, validation)
            counts[source] += 1
        return counts

    def _tokens(self, start: int, count: int, validation: bool) -> torch.Tensor:
        indices = torch.arange(start, start + count, dtype=torch.int64)
        source_numbers = torch.tensor(
            [self._source_ids[self.source_at(int(index), validation)] for index in indices], dtype=torch.int64
        )
        if self._source_tokens is not None:
            output = torch.empty(count, dtype=torch.long)
            for local_index in range(count):
                absolute = start + local_index
                source = self.source_at(absolute, validation)
                stream = self._source_tokens[source]
                phase = (absolute // self.spec.source_block_tokens) * 17 + absolute
                output[local_index] = stream[phase % stream.numel()]
            return output
        # SplitMix-like integer scrambling gives a stable nontrivial stream
        # without a giant 10M-token file. Token 0 is reserved for padding.
        domain = self.spec.seed + (0x51ED if validation else 0)
        values = indices + source_numbers * 0x9E3779B1 + domain
        values ^= values >> 30
        values *= 0xBF58476D1CE4E5B9
        values ^= values >> 27
        values *= 0x94D049BB133111EB
        values ^= values >> 31
        return (values.remainder(self.vocab_size - 1) + 1).to(torch.long)

    def batch(self, token_cursor: int, sequence_length: int | None = None, validation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        length = sequence_length or self.spec.sequence_length
        if length < 2:
            raise ValueError("sequence length must be at least 2")
        tokens = self._tokens(token_cursor, length, validation).view(1, length)
        return tokens[:, :-1], tokens[:, 1:]

    def iter_batches(
        self,
        token_budget: int,
        sequence_length: int | None = None,
        start_cursor: int = 0,
        validation: bool = False,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        length = sequence_length or self.spec.sequence_length
        cursor = start_cursor
        end = token_budget
        while cursor + length <= end:
            inputs, labels = self.batch(cursor, length, validation)
            yield cursor, inputs, labels
            cursor += length
