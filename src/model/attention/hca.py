"""Heavily Compressed Attention (HCA) branch."""

from .csa import CompressedSparseAttention


class HeavilyCompressedAttention(CompressedSparseAttention):
    """CSA with the V4-style aggressive compression ratio.

    HCA intentionally shares the same retrieval contract so the top-k and
    correctness ablations compare only the compression/retrieval regime.
    """

    pass

