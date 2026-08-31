"""Run the four real-Nemotron local architecture candidates.

The 10M-token budget is global across candidates: each of four candidates
receives the same 2.5M-token slice, with 2.4M train and 100K held-out tokens.
All candidates use the exact same composition-locked stream.  Only validation
loss/perplexity selects the winner; throughput and memory are reported as
constraints, not quality substitutes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.corpus import CompositionLockedCorpus, CorpusSpec
from model.configuration import KestrelConfig
from model.nemotron import load_real_nemotron_transplant
from training.checkpoint import CheckpointManager
from training.muon import build_muon_optimizer


OPTIONS: tuple[dict[str, object], ...] = (
    {
        "name": "target_mhc_topk512",
        "num_key_value_heads": 1,
        "hca_compression_ratio": 128,
        "index_topk": 512,
        "mhc_enabled": True,
    },
    {
        "name": "local_mhc_topk128",
        "num_key_value_heads": 1,
        "hca_compression_ratio": 128,
        "index_topk": 128,
        "mhc_enabled": True,
    },
    {
        "name": "two_kv_hca64_topk256",
        "num_key_value_heads": 2,
        "hca_compression_ratio": 64,
        "index_topk": 256,
        "mhc_enabled": True,
    },
    {
        "name": "no_mhc_topk256",
        "num_key_value_heads": 1,
        "hca_compression_ratio": 128,
        "index_topk": 256,
        "mhc_enabled": False,
    },
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_for(option: dict[str, object]) -> KestrelConfig:
    return KestrelConfig(
        num_key_value_heads=int(option["num_key_value_heads"]),
        hca_compression_ratio=int(option["hca_compression_ratio"]),
        index_topk=int(option["index_topk"]),
        mhc_enabled=bool(option["mhc_enabled"]),
        use_vision=False,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, corpus: CompositionLockedCorpus, start: int, token_budget: int, sequence_length: int, device: torch.device) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for cursor, inputs, labels in corpus.iter_batches(
        start + token_budget, sequence_length, start_cursor=start, validation=True
    ):
        del cursor
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs, labels=labels)
        count = labels.numel() - 1
        total_loss += float(output.loss.detach().cpu()) * max(count, 1)
        total_tokens += max(count, 1)
    if total_tokens == 0:
        raise RuntimeError("validation produced no complete sequence")
    return total_loss / total_tokens, total_tokens


def run_option(
    args: argparse.Namespace,
    option_index: int,
    option: dict[str, object],
    spec: CorpusSpec,
    tokenizer: object,
    device: torch.device,
) -> dict[str, object]:
    seed_all(args.seed)
    config = config_for(option)
    model, load_info = load_real_nemotron_transplant(
        config,
        model_id=args.model_id,
        revision=args.revision,
        device=device,
        load_in_4bit=True,
        compute_dtype=torch.float16,
    )
    corpus = CompositionLockedCorpus(spec, config.vocab_size, tokenizer=tokenizer)
    candidate_root = ROOT / args.checkpoint_root / f"{option_index:02d}_{option['name']}"
    manager = CheckpointManager(candidate_root, interval_steps=args.checkpoint_interval)
    optimizer = build_muon_optimizer(
        model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, weight_decay=args.weight_decay
    )
    per_candidate = args.total_tokens // len(OPTIONS)
    validation_tokens = min(args.validation_tokens, per_candidate // 4)
    train_tokens = per_candidate - validation_tokens
    sequence_length = args.sequence_length
    step = 0
    token_cursor = 0
    if args.resume and manager.latest() is not None:
        state = manager.load(manager.latest(), model, optimizer)
        step = int(state["step"])
        token_cursor = int(state.get("dataset", {}).get("token_cursor", 0))
        if state.get("dataset", {}).get("corpus_fingerprint") != spec.fingerprint():
            raise RuntimeError("checkpoint corpus fingerprint does not match current final corpus")
        print(f"resumed {option['name']} at step={step} token_cursor={token_cursor}", flush=True)
    elif args.resume:
        manager.save(
            0,
            model,
            optimizer,
            dataset_state={"token_cursor": 0, "corpus_fingerprint": spec.fingerprint()},
            metrics={"option": option, "status": "initialized_real_nemotron"},
        )

    model.train()
    start_time = time.perf_counter()
    train_loss_sum = 0.0
    train_loss_tokens = 0
    target = train_tokens
    try:
        while token_cursor < target:
            length = min(sequence_length, target - token_cursor)
            if length < 2:
                break
            inputs, labels = corpus.batch(token_cursor, length, validation=False)
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs, labels=labels)
            if output.loss is None or not torch.isfinite(output.loss):
                raise FloatingPointError(f"non-finite loss in {option['name']} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            count = max(labels.numel() - 1, 1)
            train_loss_sum += float(output.loss.detach().cpu()) * count
            train_loss_tokens += count
            token_cursor += length
            step += 1
            if manager.should_save(step):
                manager.save(
                    step,
                    model,
                    optimizer,
                    dataset_state={
                        "token_cursor": token_cursor,
                        "target_train_tokens": target,
                        "corpus_fingerprint": spec.fingerprint(),
                    },
                    metrics={"option": option, "train_loss": train_loss_sum / max(train_loss_tokens, 1)},
                )
            if manager.stop_requested:
                manager.save(
                    step,
                    model,
                    optimizer,
                    dataset_state={"token_cursor": token_cursor, "corpus_fingerprint": spec.fingerprint()},
                    metrics={"option": option, "status": "preempted_checkpoint"},
                )
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print(f"checkpointed interruption for {option['name']} at {token_cursor} tokens", flush=True)
        raise

    train_seconds = time.perf_counter() - start_time
    val_loss, val_tokens = evaluate(
        model, corpus, train_tokens, validation_tokens, sequence_length, device
    )
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
            "target_train_tokens": target,
            "validation_tokens": val_tokens,
            "corpus_fingerprint": spec.fingerprint(),
        },
        metrics={"option": option, "train_loss": train_loss_sum / max(train_loss_tokens, 1), "val_loss": val_loss},
    )
    result = {
        "option_index": option_index,
        **option,
        "corpus_name": spec.name,
        "corpus_fingerprint": spec.fingerprint(),
        "global_token_budget": args.total_tokens,
        "candidate_token_budget": per_candidate,
        "train_tokens": token_cursor,
        "validation_tokens": val_tokens,
        "train_loss": train_loss_sum / max(train_loss_tokens, 1),
        "validation_loss": val_loss,
        "validation_perplexity": math.exp(min(val_loss, 20.0)),
        "train_seconds": train_seconds,
        "train_tokens_per_second": token_cursor / max(train_seconds, 1e-9),
        "peak_vram_gib": peak_vram,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.parameter_count(trainable_only=True),
        "muon_matrix_parameters": getattr(optimizer, "matrix_parameter_count", None),
        "minimal_adamw_vector_parameters": getattr(optimizer, "vector_parameter_count", None),
        "checkpoint": str(final_checkpoint),
        "load_info": load_info.__dict__,
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/OpenReasoning-Nemotron-7B")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus-config", default="configs/data/final_corpus.yaml")
    parser.add_argument("--total-tokens", type=int, default=10_000_000)
    parser.add_argument("--validation-tokens", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--muon-lr", type=float, default=0.002)
    parser.add_argument("--adamw-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-root", default="checkpoints/real_ablations")
    parser.add_argument("--output", default="reports/ablations/real_ablation_results.json")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.total_tokens < len(OPTIONS) * 4:
        raise ValueError("total token budget is too small for four candidates")
    spec = CorpusSpec.from_yaml(ROOT / args.corpus_config)
    if spec.total_ablation_token_budget != args.total_tokens:
        raise ValueError("--total-tokens must equal total_ablation_token_budget in the final corpus config")
    if args.total_tokens % len(OPTIONS):
        raise ValueError("total token budget must divide evenly across candidates")
    device = torch.device(args.device)
    seed_all(args.seed)
    from transformers import AutoTokenizer

    # The pinned local environment may have an older Rust tokenizers parser
    # than the tokenizer.json shipped by Nemotron.  The slow Qwen tokenizer
    # uses the same immutable vocab/merges and avoids parser-version drift.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, use_fast=False
    )
    results: list[dict[str, object]] = []
    for index, option in enumerate(OPTIONS):
        print(f"[{index + 1}/{len(OPTIONS)}] {option['name']}: {option}", flush=True)
        result = run_option(args, index, option, spec, tokenizer, device)
        results.append(result)
        print(json.dumps(result, sort_keys=True, default=str), flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "protocol": vars(args),
                    "corpus": spec.to_dict(),
                    "corpus_fingerprint": spec.fingerprint(),
                    "candidate_count": len(OPTIONS),
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    winner = min(results, key=lambda item: float(item["validation_loss"]))
    final = {
        "selection": "lowest held-out validation loss; perplexity is derived",
        "global_token_budget": args.total_tokens,
        "tokens_per_candidate": args.total_tokens // len(OPTIONS),
        "corpus_fingerprint": spec.fingerprint(),
        "winner": winner,
        "continuation_contract": "continue the selected checkpoint with the same corpus fingerprint and composition",
    }
    Path("reports/ablations/final_architecture.json").write_text(
        json.dumps(final, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("WINNER=" + json.dumps(winner, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
