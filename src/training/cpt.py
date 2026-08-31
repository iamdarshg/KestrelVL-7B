"""Selective code-CPT configuration; data stays streamed and resumable."""

from dataclasses import dataclass

from data.corpus import FINAL_SOURCE_WEIGHTS


@dataclass
class CPTConfig:
    max_tokens: int = 300_000_000
    fim_rate: float = 0.35
    sequence_length: int = 8192
    learning_rate: float = 2e-5
    # This is intentionally shared with the ablation corpus and final CPT
    # manifest so the selected checkpoint can continue without a distribution
    # jump.
    source_weights: tuple[tuple[str, float], ...] = FINAL_SOURCE_WEIGHTS
