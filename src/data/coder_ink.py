"""Metadata contract for CoderInk augmentations and ambiguous glyph labels."""

from dataclasses import dataclass, field


AMBIGUOUS_GLYPHS = ("0/O", "1/l/I", "{/(", ";/:", "=/-", "</(", "[]/()")


@dataclass(frozen=True)
class CoderInkTask:
    text: str
    seed: int
    augmentations: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_glyphs: tuple[str, ...] = AMBIGUOUS_GLYPHS

