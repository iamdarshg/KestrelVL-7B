"""Run the BERT-targeted local architecture ablation.

This is deliberately a separate runner from ``run_real_ablations.py``.  BERT
is an encoder-only masked-language model, so its honest comparison protocol is
not the causal Nemotron/V4-Flash protocol.  The target is the real
``bert-base-uncased`` checkpoint; all four candidates use the same
composition-locked stream and the same deterministic 15% MLM corruption.

The global budget is 10M corpus tokens.  With four candidates this gives each
candidate 2.5M tokens: 2.4M training and 100K validation.  The validation
loss is computed only on masked tokens, while the cursor and budget count all
processed corpus tokens.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.corpus import CompositionLockedCorpus, CorpusSpec  # noqa: E402
from model.attention.mhc import ManifoldHyperConnection  # noqa: E402
from training.checkpoint import CheckpointManager  # noqa: E402
from training.muon import build_muon_optimizer  # noqa: E402


OPTIONS: tuple[dict[str, object], ...] = (
    {
        "name": "bert_global_mhc",
        "attention_window": None,
        "mhc_enabled": True,
        "description": "full bidirectional BERT attention with mHC residual mixers",
    },
    {
        "name": "bert_local128_mhc",
        "attention_window": 128,
        "mhc_enabled": True,
        "description": "bidirectional 128-token local attention with mHC residual mixers",
    },
    {
        "name": "bert_global_no_mhc",
        "attention_window": None,
        "mhc_enabled": False,
        "description": "pretrained BERT residual path without mHC",
    },
    {
        "name": "bert_local128_no_mhc",
        "attention_window": 128,
        "mhc_enabled": False,
        "description": "bidirectional 128-token local attention without mHC",
    },
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bert_classes() -> tuple[type[nn.Module], Any]:
    """Import BERT lazily so corpus/tests do not pay the transformers import cost."""
    from transformers.models.bert.modeling_bert import BertForMaskedLM, BertLayer

    return BertLayer, BertForMaskedLM


class MHCEncoderLayer(nn.Module):
    """BERT layer with mHC around the two residual updates.

    The mHC is applied to the *pre-LayerNorm residual update*, matching BERT's
    original ``LayerNorm(update + residual)`` computation.  With mHC disabled
    this is algebraically the original BERT layer, which makes the no-mHC
    candidate a meaningful pretrained baseline.
    """

    def __init__(self, config: Any, mhc_enabled: bool) -> None:
        BertLayer, _ = _bert_classes()
        # Use composition rather than subclassing so the runner remains
        # compatible with the pinned Transformers implementation.
        super().__init__()
        source = BertLayer(config)
        self.chunk_size_feed_forward = source.chunk_size_feed_forward
        self.seq_len_dim = source.seq_len_dim
        self.attention = source.attention
        self.is_decoder = source.is_decoder
        self.add_cross_attention = source.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = source.crossattention
        self.intermediate = source.intermediate
        self.output = source.output
        self.attention_mhc = ManifoldHyperConnection(
            streams=2, sinkhorn_iters=6, enabled=mhc_enabled
        )
        self.mlp_mhc = ManifoldHyperConnection(
            streams=2, sinkhorn_iters=6, enabled=mhc_enabled
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # The target checkpoint is an encoder, but retaining the standard
        # cross-attention path makes the layer safe to inspect under another
        # BERT config without silently changing its contract.
        self_attn_past = past_key_value[:2] if past_key_value is not None else None
        self_outputs = self.attention.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            self_attn_past,
            output_attentions,
        )
        attention_delta = self.attention.output.dense(self_outputs[0])
        attention_delta = self.attention.output.dropout(attention_delta)
        attention_output = self.attention.output.LayerNorm(
            self.attention_mhc(hidden_states, attention_delta)
        )

        if self.is_decoder:
            outputs = self_outputs[1:-1]
            present_key_value = self_outputs[-1]
        else:
            outputs = self_outputs[1:]

        cross_present = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError("cross attention requested but this BERT layer has none")
            cross_past = past_key_value[-2:] if past_key_value is not None else None
            cross_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_past,
                output_attentions,
            )
            attention_output = cross_outputs[0]
            outputs = outputs + cross_outputs[1:-1]
            cross_present = cross_outputs[-1]
            present_key_value = present_key_value + cross_present

        intermediate_output = self.intermediate(attention_output)
        ffn_delta = self.output.dense(intermediate_output)
        ffn_delta = self.output.dropout(ffn_delta)
        layer_output = self.output.LayerNorm(self.mlp_mhc(attention_output, ffn_delta))
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


def install_mhc_layers(model: nn.Module, enabled: bool) -> None:
    """Replace encoder layers while loading every original BERT tensor exactly."""
    _, BertForMaskedLM = _bert_classes()
    if not isinstance(model, BertForMaskedLM):
        raise TypeError("install_mhc_layers expects BertForMaskedLM")
    old_layers = list(model.bert.encoder.layer)
    new_layers: list[nn.Module] = []
    for old in old_layers:
        new = MHCEncoderLayer(model.config, enabled)
        # The wrappers have the same BERT submodule names, plus only mHC
        # parameters.  Permit exactly those new parameters to be missing;
        # unexpected/mismatched upstream BERT names still fail loudly.
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        allowed = {"attention_mhc.logits", "attention_mhc.residual_scale", "mlp_mhc.logits", "mlp_mhc.residual_scale"}
        if set(missing) != allowed or unexpected:
            raise RuntimeError(
                f"BERT layer transplant mismatch: missing={missing}, unexpected={unexpected}"
            )
        new_layers.append(new)
    model.bert.encoder.layer = nn.ModuleList(new_layers)


def freeze_lexical_io(model: nn.Module) -> None:
    """Keep BERT's tied vocabulary geometry fixed during the small screen.

    The 30,522 x 768 embedding/MLM decoder is both expensive to adapt on an
    8GB card and unsafe for Newton--Schulz Muon updates.  It is the original
    pretrained lexical interface, so freezing it is also the least invasive
    way to minimize AdamW while comparing the residual/attention choices.
    """
    for name, parameter in model.named_parameters():
        if "word_embeddings.weight" in name or "predictions.decoder.weight" in name:
            parameter.requires_grad = False


def freeze_bert_prefix(model: nn.Module, trainable_layer_start: int) -> None:
    """Freeze lower encoder blocks for the bounded local architecture screen."""
    layers = model.bert.encoder.layer  # type: ignore[attr-defined]
    if not 0 <= trainable_layer_start <= len(layers):
        raise ValueError(
            f"trainable_layer_start must be in [0, {len(layers)}], got {trainable_layer_start}"
        )
    for index, layer in enumerate(layers):
        if index < trainable_layer_start:
            for parameter in layer.parameters():
                parameter.requires_grad = False


def local_attention_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    if window < 1:
        raise ValueError("attention window must be positive")
    positions = torch.arange(length, device=device)
    return (positions[:, None] - positions[None, :]).abs().le(window).to(torch.float32).unsqueeze(0)


def _mask_generator(seed: int, cursor: int, validation: bool) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    domain = seed + (0x5A17 if validation else 0)
    generator.manual_seed((domain + cursor * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF)
    return generator


def make_mlm_batch(
    corpus: CompositionLockedCorpus,
    cursor: int,
    length: int,
    tokenizer: Any,
    seed: int,
    validation: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create deterministic BERT masking without consuming global RNG state."""
    if length < 2:
        raise ValueError("BERT sequence length must be at least 2")
    # CompositionLockedCorpus.batch returns a shifted causal pair.  The first
    # member is exactly the requested contiguous token block, so no token is
    # dropped from the global budget here.
    input_ids = corpus._tokens(cursor, length, validation).view(1, length).clone()  # noqa: SLF001
    original = input_ids.clone()
    generator = _mask_generator(seed, cursor, validation)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    special_ids = tuple(int(value) for value in getattr(tokenizer, "all_special_ids", ()))
    if special_ids:
        special = torch.tensor(special_ids, dtype=torch.long)
        eligible &= ~torch.isin(input_ids.cpu(), special).to(input_ids.device)
    masked = (torch.rand(input_ids.shape, generator=generator) < 0.15) & eligible.cpu()
    masked = masked.to(input_ids.device)
    labels = torch.full_like(input_ids, -100)
    labels[masked] = original[masked]
    replacement = torch.rand(input_ids.shape, generator=generator)
    mask_token_id = int(tokenizer.mask_token_id)
    replace_with_mask = masked & (replacement < 0.8)
    replace_with_random = masked & (replacement >= 0.9)
    input_ids[replace_with_mask] = mask_token_id
    if replace_with_random.any():
        random_ids = torch.randint(
            low=0,
            high=int(tokenizer.vocab_size),
            size=input_ids.shape,
            generator=generator,
            dtype=torch.long,
        ).to(input_ids.device)
        input_ids[replace_with_random] = random_ids[replace_with_random]
    return input_ids, labels, int(masked.sum().item())


def batch_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if output.loss is None:
        raise RuntimeError("BERT did not return an MLM loss")
    count = int(labels.ne(-100).sum().item())
    if count == 0:
        raise RuntimeError("deterministic MLM batch contained no masked tokens")
    return output.loss, count


def target_config_digest(model_id: str, revision: str | None) -> tuple[dict[str, object], str]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    payload = config.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def retain_only_best_archive(results: list[dict[str, object]], archive_root: Path | None) -> None:
    if archive_root is None or not results:
        return
    archive_root = archive_root.resolve()
    best_index = min(range(len(results)), key=lambda idx: float(results[idx]["validation_loss"]))
    for index, result in enumerate(results):
        if index == best_index or not result.get("checkpoint"):
            continue
        candidate = Path(str(result["checkpoint"]))
        if candidate.resolve().parent != archive_root or not candidate.is_dir():
            raise RuntimeError(f"refusing to prune BERT checkpoint outside archive root: {candidate}")
        shutil.rmtree(candidate)
        result["checkpoint"] = None
        result["checkpoint_retained"] = False
    results[best_index]["checkpoint_retained"] = True


def run_option(
    args: argparse.Namespace,
    option_index: int,
    option: dict[str, object],
    spec: CorpusSpec,
    tokenizer: Any,
    device: torch.device,
    target_config: dict[str, object],
    target_digest: str,
) -> dict[str, object]:
    _, BertForMaskedLM = _bert_classes()
    seed_all(args.seed + option_index)
    compute_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = BertForMaskedLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=compute_dtype if device.type == "cuda" else torch.float32,
        local_files_only=Path(args.model_id).is_dir(),
    )
    install_mhc_layers(model, bool(option["mhc_enabled"]))
    freeze_lexical_io(model)
    freeze_bert_prefix(model, args.trainable_layer_start)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.config.use_cache = False

    max_positions = int(getattr(model.config, "max_position_embeddings", 512))
    if args.sequence_length > max_positions:
        raise ValueError(
            f"BERT target supports at most {max_positions} positions, got {args.sequence_length}; "
            "use --sequence-length 512 or explicitly extend the target before training"
        )
    corpus = CompositionLockedCorpus(spec, int(model.config.vocab_size), tokenizer=tokenizer)
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    candidate_root = checkpoint_root / f"{option_index:02d}_{option['name']}"
    manager = CheckpointManager(
        candidate_root, interval_steps=args.checkpoint_interval, max_checkpoints=args.max_checkpoints
    )
    optimizer = build_muon_optimizer(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        muon_max_matrix_dimension=args.muon_max_matrix_dimension,
    )
    per_candidate = args.total_tokens // len(OPTIONS)
    validation_tokens = min(args.validation_tokens, per_candidate // 4)
    train_tokens = per_candidate - validation_tokens
    step = 0
    token_cursor = 0
    if args.resume and manager.latest() is not None:
        state = manager.load(manager.latest(), model, optimizer)
        step = int(state["step"])
        token_cursor = int(state.get("dataset", {}).get("token_cursor", 0))
        if state.get("dataset", {}).get("corpus_fingerprint") != spec.fingerprint():
            raise RuntimeError("BERT checkpoint corpus fingerprint does not match final corpus")
        print(f"resumed {option['name']} at step={step} token_cursor={token_cursor}", flush=True)

    window = option["attention_window"]
    attention_mask = None
    if window is not None:
        attention_mask = local_attention_mask(args.sequence_length, int(window), device)

    model.train()
    start_time = time.perf_counter()
    train_loss_sum = 0.0
    train_masked = 0
    try:
        while token_cursor < train_tokens:
            length = min(args.sequence_length, train_tokens - token_cursor)
            if length < 2:
                break
            input_ids, labels, _ = make_mlm_batch(
                corpus, token_cursor, length, tokenizer, args.seed + option_index, False
            )
            input_ids, labels = input_ids.to(device), labels.to(device)
            batch_mask = attention_mask if attention_mask is not None and length == args.sequence_length else None
            loss, masked_count = batch_loss(model, input_ids, labels, batch_mask, compute_dtype, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite BERT loss in {option['name']} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * masked_count
            train_masked += masked_count
            token_cursor += length
            step += 1
            if step == 1 or step % args.log_interval == 0:
                current_loss = train_loss_sum / max(train_masked, 1)
                memory = (
                    torch.cuda.memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"{option['name']} step={step} tokens={token_cursor}/{train_tokens} "
                    f"masked_loss={current_loss:.5f} vram_gib={memory:.3f}",
                    flush=True,
                )
            if manager.should_save(step):
                manager.save(
                    step,
                    model,
                    optimizer,
                    dataset_state={
                        "token_cursor": token_cursor,
                        "target_train_tokens": train_tokens,
                        "corpus_fingerprint": spec.fingerprint(),
                    },
                    metrics={
                        "option": option,
                        "train_loss": train_loss_sum / max(train_masked, 1),
                        "target": "bert",
                    },
                )
            if manager.stop_requested:
                manager.save(
                    step,
                    model,
                    optimizer,
                    dataset_state={"token_cursor": token_cursor, "corpus_fingerprint": spec.fingerprint()},
                    metrics={"option": option, "status": "preempted_checkpoint", "target": "bert"},
                )
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print(f"checkpointed interruption for {option['name']} at {token_cursor} tokens", flush=True)
        raise

    model.eval()
    validation_loss_sum = 0.0
    validation_masked = 0
    validation_cursor = train_tokens
    validation_end = train_tokens + validation_tokens
    with torch.no_grad():
        while validation_cursor < validation_end:
            length = min(args.sequence_length, validation_end - validation_cursor)
            if length < 2:
                break
            input_ids, labels, _ = make_mlm_batch(
                corpus, validation_cursor, length, tokenizer, args.seed + option_index, True
            )
            input_ids, labels = input_ids.to(device), labels.to(device)
            batch_mask = attention_mask if attention_mask is not None and length == args.sequence_length else None
            loss, masked_count = batch_loss(model, input_ids, labels, batch_mask, compute_dtype, device)
            validation_loss_sum += float(loss.detach().cpu()) * masked_count
            validation_masked += masked_count
            validation_cursor += length
    if validation_masked == 0:
        raise RuntimeError("validation produced no masked tokens")
    validation_loss = validation_loss_sum / validation_masked
    train_seconds = time.perf_counter() - start_time
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        peak_vram = 0.0
    final_checkpoint = manager.save(
        step,
        model,
        optimizer,
        dataset_state={
            "token_cursor": token_cursor,
            "target_train_tokens": train_tokens,
            "validation_tokens": validation_tokens,
            "validation_masked_tokens": validation_masked,
            "corpus_fingerprint": spec.fingerprint(),
        },
        metrics={
            "option": option,
            "train_loss": train_loss_sum / max(train_masked, 1),
            "validation_loss": validation_loss,
            "target": "bert",
        },
    )
    optimizer_snapshot = final_checkpoint / "optimizer.pt"
    if optimizer_snapshot.exists():
        optimizer_snapshot.unlink()
    archive_root = Path(args.final_archive_root) if args.final_archive_root else None
    if archive_root is not None:
        if not archive_root.is_absolute():
            archive_root = ROOT / archive_root
        archive_path = archive_root / f"{option_index:02d}_{option['name']}"
        archive_root.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            raise FileExistsError(f"refusing to overwrite archived BERT candidate {archive_path}")
        shutil.move(str(final_checkpoint), str(archive_path))
        final_checkpoint = archive_path
    result: dict[str, object] = {
        "target": "bert",
        "target_model_id": args.model_id,
        "target_revision": args.revision,
        "target_config_sha256": target_digest,
        "target_config": target_config,
        "option_index": option_index,
        **option,
        "corpus_name": spec.name,
        "corpus_fingerprint": spec.fingerprint(),
        "global_token_budget": args.total_tokens,
        "candidate_token_budget": per_candidate,
        "train_tokens": token_cursor,
        "validation_tokens": validation_cursor - train_tokens,
        "validation_masked_tokens": validation_masked,
        "train_loss": train_loss_sum / max(train_masked, 1),
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "train_seconds": train_seconds,
        "train_tokens_per_second": token_cursor / max(train_seconds, 1e-9),
        "gpu_hours": train_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "peak_vram_gib": peak_vram,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "muon_matrix_parameters": getattr(optimizer, "matrix_parameter_count", None),
        "minimal_adamw_vector_parameters": getattr(optimizer, "vector_parameter_count", None),
        "adamw_fallback_matrix_parameters": getattr(optimizer, "adamw_fallback_matrix_parameter_count", None),
        "compute_dtype": str(compute_dtype),
        "trainable_layer_start": args.trainable_layer_start,
        "sequence_length": args.sequence_length,
        "checkpoint": str(final_checkpoint),
        "optimizer_snapshot_retained": False,
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="bert-base-uncased")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus-config", default="configs/data/final_corpus.yaml")
    parser.add_argument("--total-tokens", type=int, default=10_000_000)
    parser.add_argument("--validation-tokens", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--muon-lr", type=float, default=0.002)
    parser.add_argument("--adamw-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--muon-max-matrix-dimension",
        type=int,
        default=4096,
        help="route larger trainable matrices to AdamW; the BERT vocabulary matrix is frozen",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trainable-layer-start",
        type=int,
        default=8,
        help="train only this encoder suffix; default 8 means the top four BERT layers",
    )
    parser.add_argument("--checkpoint-root", default="checkpoints/bert_ablations")
    parser.add_argument("--final-archive-root", default="checkpoints/bert_ablations_best")
    parser.add_argument("--output", default="reports/ablations/bert_ablation_results.json")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.total_tokens < len(OPTIONS) * 4 or args.total_tokens % len(OPTIONS):
        raise ValueError("total token budget must be divisible across four BERT candidates")
    spec = CorpusSpec.from_yaml(ROOT / args.corpus_config)
    if spec.total_ablation_token_budget != args.total_tokens:
        raise ValueError("--total-tokens must equal total_ablation_token_budget in final_corpus.yaml")
    device = torch.device(args.device)
    seed_all(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        use_fast=False,
        local_files_only=Path(args.model_id).is_dir(),
    )
    target_config, target_digest = target_config_digest(args.model_id, args.revision)
    reference_path = ROOT / "references" / "bert_base_uncased_config.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if not reference_path.exists():
        reference_path.write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "config_sha256": target_digest,
                    "config": target_config,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    results: list[dict[str, object]] = []
    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("target") != "bert":
            raise RuntimeError(f"resume report target mismatch: {payload.get('target')!r}")
        if int(payload.get("global_token_budget", -1)) != args.total_tokens:
            raise RuntimeError("resume report global token budget mismatch")
        if payload.get("corpus_fingerprint") != spec.fingerprint():
            raise RuntimeError("resume report corpus fingerprint mismatch")
        if payload.get("target_config_sha256") != target_digest:
            raise RuntimeError("resume report target config mismatch")
        for prior in payload.get("results", []):
            if not isinstance(prior, dict):
                raise RuntimeError("resume report contains a non-object result")
            if prior.get("target") != "bert" or prior.get("corpus_fingerprint") != spec.fingerprint():
                raise RuntimeError("resume report contains an incompatible candidate")
            results.append(prior)
        if results:
            print(f"loaded {len(results)} completed BERT candidate result(s) from {output_path}", flush=True)
    archive_root = Path(args.final_archive_root)
    if not archive_root.is_absolute():
        archive_root = ROOT / archive_root
    for index, option in enumerate(OPTIONS):
        completed = next((prior for prior in results if prior.get("name") == option["name"]), None)
        if completed is not None:
            print(f"[{index + 1}/{len(OPTIONS)}] {option['name']}: completed result loaded; skipping retraining", flush=True)
            continue
        print(f"[{index + 1}/{len(OPTIONS)}] {option['name']}: {option}", flush=True)
        result = run_option(
            args, index, option, spec, tokenizer, device, target_config, target_digest
        )
        results.append(result)
        retain_only_best_archive(results, archive_root)
        print(json.dumps(result, sort_keys=True, default=str), flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "protocol": vars(args),
                    "target": "bert",
                    "target_config_sha256": target_digest,
                    "corpus": spec.to_dict(),
                    "corpus_fingerprint": spec.fingerprint(),
                    "candidate_count": len(OPTIONS),
                    "global_token_budget": args.total_tokens,
                    "per_candidate_token_budget": args.total_tokens // len(OPTIONS),
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    winner = min(results, key=lambda item: float(item["validation_loss"]))
    selection = {
        "target": "bert",
        "winner": winner,
        "selection_metric": "held-out masked-language-model validation loss",
        "note": "This BERT target screen is not comparable to the causal Nemotron/V4 screen.",
    }
    (ROOT / "reports/ablations/bert_target_selection.json").write_text(
        json.dumps(selection, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
