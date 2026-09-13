"""Shared contract for the heterogeneous Qwen3.5 CED runtime (issues #2/#3/#4).

This module pins the source-model identities and architecture metadata so the
CED runtime, bridge, and distillation harness share one source of truth.
It is intentionally dependency-light (stdlib only) so configs can be validated
without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ENCODER_MODEL_ID = "Qwen/Qwen3.5-2B-Base"
DECODER_MODEL_ID = "Qwen/Qwen3.5-9B"

ENCODER_HIDDEN_SIZE = 2048
ENCODER_LAYERS = 24
DECODER_HIDDEN_SIZE = 4096
DECODER_LAYERS = 32
# Two distinct vocab numbers (L4 smoke 2026-09-13 proved the distinction):
# - TOKENIZER_VOCAB_SIZE: len(tokenizer), identical both sides (verified
#   added-vocab diff is empty);
# - EMBEDDING_VOCAB_SIZE: padded text_config vocab_size / LM-head rows.
SHARED_VOCAB_SIZE = 248320  # embedding rows (kept as the arch-level name)
TOKENIZER_VOCAB_SIZE = 248077
EMBEDDING_VOCAB_SIZE = 248320

# Numerical tolerance for zero-gate decoder recovery checks (logit space).
ZERO_GATE_LOGIT_TOL = 1e-5


@dataclass(frozen=True)
class CEDSourceConfig:
    """Reproducible source-model contract for the CED student.

    Revisions pin exact pretrained checkpoints. Empty revision means
    "unpinned" and must fail loudly when strict reproducibility is required.
    """

    encoder_model_id: str = ENCODER_MODEL_ID
    encoder_revision: str = ""
    decoder_model_id: str = DECODER_MODEL_ID
    decoder_revision: str = ""
    encoder_hidden_size: int = ENCODER_HIDDEN_SIZE
    encoder_layers: int = ENCODER_LAYERS
    decoder_hidden_size: int = DECODER_HIDDEN_SIZE
    decoder_layers: int = DECODER_LAYERS
    vocab_size: int = SHARED_VOCAB_SIZE
    tokenizer_vocab_size: int = TOKENIZER_VOCAB_SIZE
    expected_special_token_ids: dict[str, int] = field(default_factory=dict)

    def validate_architecture(self) -> None:
        if self.encoder_hidden_size != ENCODER_HIDDEN_SIZE:
            raise ValueError(
                f"encoder hidden size must be {ENCODER_HIDDEN_SIZE}, "
                f"got {self.encoder_hidden_size}"
            )
        if self.encoder_layers != ENCODER_LAYERS:
            raise ValueError(f"encoder layers must be {ENCODER_LAYERS}, got {self.encoder_layers}")
        if self.decoder_hidden_size != DECODER_HIDDEN_SIZE:
            raise ValueError(
                f"decoder hidden size must be {DECODER_HIDDEN_SIZE}, "
                f"got {self.decoder_hidden_size}"
            )
        if self.decoder_layers != DECODER_LAYERS:
            raise ValueError(f"decoder layers must be {DECODER_LAYERS}, got {self.decoder_layers}")
        if self.vocab_size != SHARED_VOCAB_SIZE:
            raise ValueError(f"vocab size must be {SHARED_VOCAB_SIZE}, got {self.vocab_size}")
        if self.encoder_hidden_size == self.decoder_hidden_size:
            raise ValueError(
                "heterogeneous CED requires mismatched hidden sizes; "
                "equal sizes suggest a config error, not a reshape opportunity"
            )

    def require_pinned_revisions(self) -> None:
        missing = [
            name
            for name, value in (
                ("encoder_revision", self.encoder_revision),
                ("decoder_revision", self.decoder_revision),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"unpinned source revisions: {', '.join(missing)}")

    def check_tokenizer_compatibility(
        self,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        encoder_special_ids: dict[str, int] | None = None,
        decoder_special_ids: dict[str, int] | None = None,
    ) -> None:
        """Fail loudly on any tokenizer/special-token divergence.

        Tokenizer lengths are checked against ``tokenizer_vocab_size``
        (248077), NOT the padded embedding size — the L4 smoke test proved
        these differ. Use ``validate_architecture`` for embedding rows.
        """
        if self.tokenizer_vocab_size != TOKENIZER_VOCAB_SIZE:
            raise ValueError(
                f"contract tokenizer vocab {self.tokenizer_vocab_size} != {TOKENIZER_VOCAB_SIZE}"
            )
        if encoder_vocab_size != self.tokenizer_vocab_size:
            raise ValueError(
                f"encoder vocab {encoder_vocab_size} != contract {self.tokenizer_vocab_size}"
            )
        if decoder_vocab_size != self.tokenizer_vocab_size:
            raise ValueError(
                f"decoder vocab {decoder_vocab_size} != contract {self.tokenizer_vocab_size}"
            )
        if encoder_vocab_size != decoder_vocab_size:
            raise ValueError("encoder/decoder vocab sizes diverge")
        enc_special = encoder_special_ids or {}
        dec_special = decoder_special_ids or {}
        for key in set(enc_special) | set(dec_special) | set(self.expected_special_token_ids):
            expected = self.expected_special_token_ids.get(key)
            enc_value = enc_special.get(key, expected)
            dec_value = dec_special.get(key, expected)
            if enc_value != dec_value:
                raise ValueError(
                    f"special token {key!r} diverges: encoder={enc_value} decoder={dec_value}"
                )
            if expected is not None and (enc_value != expected or dec_value != expected):
                raise ValueError(
                    f"special token {key!r} does not match contract {expected}: "
                    f"encoder={enc_value} decoder={dec_value}"
                )


def assert_no_hidden_size_splice(encoder_dim: int, decoder_dim: int) -> None:
    """Guard against faking hidden-size compatibility via truncation/padding."""
    if encoder_dim == ENCODER_HIDDEN_SIZE and decoder_dim == DECODER_HIDDEN_SIZE:
        return
    raise ValueError(
        f"explicit hidden-size contract violated: encoder_dim={encoder_dim} "
        f"(expected {ENCODER_HIDDEN_SIZE}), decoder_dim={decoder_dim} "
        f"(expected {DECODER_HIDDEN_SIZE}). "
        "Use EncoderMemoryBridge for mapping; never truncate/pad."
    )
