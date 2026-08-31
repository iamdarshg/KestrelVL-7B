"""Metadata contract for the deterministic CoderVision renderer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoderVisionTask:
    source_text: str
    ui: str
    viewport: tuple[int, int]
    theme: str
    dpi: int
    seed: int
    target: str

