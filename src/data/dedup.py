"""Deduplication interfaces; expensive optional backends remain opt-in."""

import hashlib
import re


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def token_shingles(text: str, width: int = 5) -> set[str]:
    tokens = text.split()
    return {" ".join(tokens[i : i + width]) for i in range(max(0, len(tokens) - width + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0

