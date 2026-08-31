"""Repository-aware Fill-In-the-Middle transformation."""

import random


def make_fim(prefix: str, middle: str, suffix: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    if rng.random() < 0.5:
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"


def should_fim(index: int, rate: float = 0.35) -> bool:
    # Deterministic sampling makes mixture manifests reproducible.
    return ((index * 2654435761) & 0xFFFFFFFF) / 2**32 < rate

