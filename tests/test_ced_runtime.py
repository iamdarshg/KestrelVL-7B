"""Tests for the heterogeneous CED runtime doubles (issue #3).

CPU-only, tiny, fast. No network, no HF downloads.
"""

import hashlib

import pytest
import torch

from model.ced_config import (
    CEDSourceConfig,
    DECODER_HIDDEN_SIZE,
    EMBEDDING_VOCAB_SIZE,
    ENCODER_HIDDEN_SIZE,
    TOKENIZER_VOCAB_SIZE,
    ZERO_GATE_LOGIT_TOL,
    assert_no_hidden_size_splice,
)


TINY_VOCAB = 128
TINY_ENC_DIM = 32
TINY_DEC_DIM = 48
TINY_LAYERS = 2


def make_runtime(**kwargs):
    from model.ced import CEDRuntime, ExternalMemoryHook, TinyAutoregressiveDecoder, TinyCausalEncoder

    torch.manual_seed(0)
    encoder = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=TINY_LAYERS, vocab_size=TINY_VOCAB)
    decoder = TinyAutoregressiveDecoder(
        hidden_dim=TINY_DEC_DIM, num_layers=TINY_LAYERS, vocab_size=TINY_VOCAB
    )
    hook = ExternalMemoryHook(decoder_dim=TINY_DEC_DIM, encoder_dim=TINY_ENC_DIM)
    return CEDRuntime(encoder, decoder, hook, **kwargs)


def random_ids(batch=2, seq=8, vocab=TINY_VOCAB):
    return torch.randint(0, vocab, (batch, seq))


# --- tokenizer compat -------------------------------------------------------


def test_tokenizer_compat_matching_passes():
    cfg = CEDSourceConfig(
        expected_special_token_ids={"bos": 1, "eos": 2},
    )
    cfg.check_tokenizer_compatibility(
        TOKENIZER_VOCAB_SIZE,
        TOKENIZER_VOCAB_SIZE,
        {"bos": 1, "eos": 2},
        {"bos": 1, "eos": 2},
    )


def test_tokenizer_compat_divergent_vocab_raises():
    cfg = CEDSourceConfig()
    with pytest.raises(ValueError):
        cfg.check_tokenizer_compatibility(TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE - 1)
    with pytest.raises(ValueError):
        cfg.check_tokenizer_compatibility(TOKENIZER_VOCAB_SIZE - 1, TOKENIZER_VOCAB_SIZE)


def test_tokenizer_compat_divergent_special_ids_raise():
    cfg = CEDSourceConfig(expected_special_token_ids={"bos": 1})
    with pytest.raises(ValueError):
        cfg.check_tokenizer_compatibility(
            TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE, {"bos": 1}, {"bos": 99}
        )


def test_tokenizer_vocab_differs_from_embedding_vocab_by_design():
    # L4 smoke 2026-09-13: len(tokenizer) == 248077 both sides, while the
    # padded LM-head has 248320 rows. The compat check must use the former.
    assert TOKENIZER_VOCAB_SIZE == 248077
    assert EMBEDDING_VOCAB_SIZE == 248320
    cfg = CEDSourceConfig()
    with pytest.raises(ValueError, match="contract 248077"):
        cfg.check_tokenizer_compatibility(EMBEDDING_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)


# --- hidden width contract --------------------------------------------------


def test_contract_fn_passes_for_real_dims():
    assert assert_no_hidden_size_splice(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE) is None


def test_contract_fn_raises_on_mismatch():
    with pytest.raises(ValueError):
        assert_no_hidden_size_splice(ENCODER_HIDDEN_SIZE, ENCODER_HIDDEN_SIZE)
    with pytest.raises(ValueError):
        assert_no_hidden_size_splice(16, 16)


def test_runtime_assert_hidden_contract_raises_for_tiny_dims():
    rt = make_runtime()
    with pytest.raises(ValueError):
        rt.assert_hidden_contract()


def test_runtime_constructor_rejects_dim_mismatch():
    from model.ced import CEDRuntime, ExternalMemoryHook, TinyAutoregressiveDecoder, TinyCausalEncoder

    torch.manual_seed(0)
    enc = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    dec = TinyAutoregressiveDecoder(hidden_dim=TINY_DEC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    hook = ExternalMemoryHook(decoder_dim=TINY_DEC_DIM, encoder_dim=TINY_ENC_DIM)
    with pytest.raises(ValueError):
        CEDRuntime(enc, dec, hook, encoder_dim=9999, decoder_dim=TINY_DEC_DIM)


# --- freeze / unfreeze -------------------------------------------------------


def test_independent_freeze_unfreeze():
    rt = make_runtime()
    for p in rt.encoder.parameters():
        assert p.requires_grad
    for p in rt.decoder.parameters():
        assert p.requires_grad

    rt.freeze_encoder()
    assert all(not p.requires_grad for p in rt.encoder.parameters())
    assert all(p.requires_grad for p in rt.decoder.parameters())

    rt.unfreeze_encoder()
    assert all(p.requires_grad for p in rt.encoder.parameters())

    rt.freeze_decoder()
    assert all(not p.requires_grad for p in rt.decoder.parameters())
    assert all(p.requires_grad for p in rt.encoder.parameters())

    rt.unfreeze_decoder()
    assert all(p.requires_grad for p in rt.decoder.parameters())


def test_trainable_parameter_counts():
    rt = make_runtime()
    counts = rt.trainable_parameter_counts()
    assert set(counts) >= {"encoder", "decoder", "hook", "total"}
    total = sum(p.numel() for p in rt.encoder.parameters())
    assert counts["encoder"] == total
    assert counts["total"] == counts["encoder"] + counts["decoder"] + counts["hook"]
    rt.freeze_encoder()
    counts = rt.trainable_parameter_counts()
    assert counts["encoder"] == 0
    assert counts["decoder"] > 0


# --- zero gate equivalence ---------------------------------------------------


def test_hook_gate_init_exactly_zero():
    from model.ced import ExternalMemoryHook

    hook = ExternalMemoryHook(decoder_dim=8, encoder_dim=8)
    assert hook.gate.item() == 0.0


def test_hook_identity_when_disabled_bit_exact():
    from model.ced import ExternalMemoryHook

    torch.manual_seed(1)
    hook = ExternalMemoryHook(decoder_dim=8, encoder_dim=8, enabled=False)
    hidden = torch.randn(2, 5, 8)
    memory = torch.randn(2, 7, 8)
    out = hook(hidden, memory)
    assert torch.equal(out, hidden)
    # explicit gate_scale=0 path is also bit-exact
    hook.set_enabled(True)
    out2 = hook(hidden, memory, gate_scale=0.0)
    assert torch.equal(out2, hidden)
    assert torch.max(torch.abs(out2 - hidden)).item() == 0.0


def test_hook_zero_gate_enabled_matches_input_within_tol():
    from model.ced import ExternalMemoryHook

    torch.manual_seed(2)
    hook = ExternalMemoryHook(decoder_dim=8, encoder_dim=8, enabled=True)
    assert hook.gate.item() == 0.0
    hidden = torch.randn(2, 5, 8)
    memory = torch.randn(2, 7, 8)
    out = hook(hidden, memory)
    assert torch.allclose(out, hidden, atol=ZERO_GATE_LOGIT_TOL, rtol=0)


def test_runtime_decode_without_memory_matches_standalone_decoder():
    rt = make_runtime()
    rt.set_memory_enabled(True)
    ids = random_ids()
    with torch.no_grad():
        expected = rt.decoder(ids)
        got = rt.decode_without_memory(ids)
    assert torch.allclose(got, expected, atol=ZERO_GATE_LOGIT_TOL, rtol=0)


def test_runtime_zero_gate_decode_with_memory_matches_standalone():
    rt = make_runtime()
    rt.set_memory_enabled(True)
    assert rt.memory_hook.gate.item() == 0.0
    ids = random_ids()
    enc = rt.encode_prompt(random_ids(seq=6))
    with torch.no_grad():
        expected = rt.decoder(ids)
        got = rt.decode_with_memory(ids, enc)
    assert torch.allclose(got, expected, atol=ZERO_GATE_LOGIT_TOL, rtol=0)


# --- phase separation --------------------------------------------------------


def test_decode_without_memory_does_not_call_encoder():
    rt = make_runtime()
    calls = {"n": 0}
    orig_forward = rt.encoder.forward

    def spy(input_ids, *args, **kwargs):
        calls["n"] += 1
        return orig_forward(input_ids, *args, **kwargs)

    rt.encoder.forward = spy
    ids = random_ids()
    with torch.no_grad():
        rt.decode_without_memory(ids)
    assert calls["n"] == 0
    with torch.no_grad():
        rt.encode_prompt(ids)
    assert calls["n"] == 1


def test_encode_decode_phase_types():
    from model.ced import PromptEncoding

    rt = make_runtime()
    ids = random_ids()
    with torch.no_grad():
        enc = rt.encode_prompt(ids)
    assert isinstance(enc, PromptEncoding)
    assert enc.memory.shape == (ids.shape[0], ids.shape[1], TINY_ENC_DIM)
    with torch.no_grad():
        logits = rt.decode_with_memory(ids, enc)
    assert logits.shape == (ids.shape[0], ids.shape[1], TINY_VOCAB)


# --- strict load / no random reinit ------------------------------------------


def _fresh_encoder():
    from model.ced import TinyCausalEncoder

    torch.manual_seed(0)
    return TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=TINY_LAYERS, vocab_size=TINY_VOCAB)


def test_strict_load_missing_keys_raises():
    enc = _fresh_encoder()
    full = {k: v.clone() for k, v in enc.state_dict().items()}
    partial = dict(list(full.items())[:-1])
    assert len(partial) < len(full)
    with pytest.raises(ValueError):
        enc.load_source_state(partial, strict=True)


def test_strict_load_unexpected_keys_raises():
    enc = _fresh_encoder()
    full = {k: v.clone() for k, v in enc.state_dict().items()}
    full["bogus.unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError):
        enc.load_source_state(full, strict=True)


def test_strict_load_exact_tensors_and_key_tracking():
    from model.ced import TinyCausalEncoder

    torch.manual_seed(0)
    enc = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    ckpt = {k: torch.randn_like(v) for k, v in enc.state_dict().items()}
    enc.load_source_state(ckpt, strict=True)
    for k, v in enc.state_dict().items():
        assert torch.equal(v, ckpt[k])
    assert set(enc.inherited_keys) == set(ckpt.keys())
    assert enc.new_keys == []


def test_fingerprint_stable_across_identical_loads():
    from model.ced import TinyCausalEncoder

    torch.manual_seed(0)
    enc_a = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    ckpt = {k: torch.randn_like(v) for k, v in enc_a.state_dict().items()}

    torch.manual_seed(1234)
    enc_b = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    torch.manual_seed(9999)
    enc_c = TinyCausalEncoder(hidden_dim=TINY_ENC_DIM, num_layers=1, vocab_size=TINY_VOCAB)
    enc_b.load_source_state({k: v.clone() for k, v in ckpt.items()}, strict=True)
    enc_c.load_source_state({k: v.clone() for k, v in ckpt.items()}, strict=True)
    assert enc_b.parameter_checksums() == enc_c.parameter_checksums()


def test_parameter_checksums_are_sha256():
    rt = make_runtime()
    sums = rt.parameter_checksums()
    assert len(sums) > 0
    for name, digest in sums.items():
        if name == "_combined":
            continue
        assert isinstance(digest, str) and len(digest) == 64
        int(digest, 16)
    # combined fingerprint is deterministic
    assert sums["_combined"] == rt.parameter_checksums()["_combined"]
    # spot-check one entry against manual hash
    first = next(n for n in sums if n != "_combined")
    for prefix, mod in (("encoder.", rt.encoder), ("decoder.", rt.decoder), ("hook.", rt.memory_hook)):
        if first.startswith(prefix):
            param = dict(mod.named_parameters())[first[len(prefix):]]
            manual = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
            assert manual == sums[first]
            break


# --- source config validation -------------------------------------------------


def test_validate_sources_pinned_passes():
    rt = make_runtime(
        source_config=CEDSourceConfig(encoder_revision="abc123", decoder_revision="def456")
    )
    assert rt.validate_sources(strict_revisions=True) is None


def test_validate_sources_unpinned_strict_raises():
    rt = make_runtime(source_config=CEDSourceConfig())
    with pytest.raises(ValueError):
        rt.validate_sources(strict_revisions=True)


def test_validate_sources_unpinned_nonstrict_passes():
    rt = make_runtime(source_config=CEDSourceConfig())
    assert rt.validate_sources(strict_revisions=False) is None


def test_set_memory_enabled_toggles_hook():
    rt = make_runtime()
    rt.set_memory_enabled(False)
    assert rt.memory_hook.enabled is False
    rt.set_memory_enabled(True)
    assert rt.memory_hook.enabled is True
