"""Exact and lightweight near-duplicate/benchmark exclusion checks."""

import hashlib
import re
from dataclasses import dataclass, field


def normalize_code(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#[^\n]*", "", text)
    return text.strip().lower()


@dataclass
class ContaminationIndex:
    held_out_ids: set[str] = field(default_factory=set)
    exact_hashes: set[str] = field(default_factory=set)
    token_signatures: dict[str, str] = field(default_factory=dict)

    def add_held_out(self, sample_id: str) -> None:
        self.held_out_ids.add(sample_id)

    def add(self, sample_id: str, text: str) -> None:
        normalized = normalize_code(text)
        self.exact_hashes.add(hashlib.sha256(normalized.encode()).hexdigest())
        self.token_signatures[sample_id] = normalized[:512]

    def check(self, sample_id: str, text: str) -> str:
        if sample_id in self.held_out_ids:
            return "benchmark_holdout"
        digest = hashlib.sha256(normalize_code(text).encode()).hexdigest()
        if digest in self.exact_hashes:
            return "exact_duplicate"
        return "clean"

