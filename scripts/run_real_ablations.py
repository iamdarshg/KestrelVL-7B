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
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def retain_only_best_archive(results: list[dict[str, object]], archive_root: str | None) -> None:
    """Keep the current best model-only final when local disk is constrained."""
    if not archive_root or not results:
        return
    root = Path(archive_root).resolve()
    best_index = min(range(len(results)), key=lambda idx: float(results[idx]["validation_loss"]))
    for idx, result in enumerate(results):
        if idx == best_index or not result.get("checkpoint"):
            continue
        candidate = Path(str(result["checkpoint"]))
        if candidate.resolve().parent != root or not candidate.is_dir():
            raise RuntimeError(f"refusing to prune checkpoint outside archive root: {candidate}")
        shutil.rmtree(candidate)
        result["checkpoint"] = None
        result["checkpoint_retained"] = False
    results[best_index]["checkpoint_retained"] = True


def loss_for_batch(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Score the already-shifted corpus batch without dropping another token.

    ``CompositionLockedCorpus.batch`` returns input/target pairs with equal
    length.  The real model's public ``labels=`` API follows Transformers'
    same-length convention and shifts internally, so using it here would
    shift twice.  The ablation runner therefore computes CE directly and
    counts every target token in the fixed budget.
    """
    output = model(inputs)
    return F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
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


def validate_local_model_files(model_id: str) -> None:
    """Refuse to spend GPU time against an incomplete local checkpoint."""
    model_dir = Path(model_id)
    if not model_dir.is_dir():
        return
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"local model is missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in expected if not (model_dir / name).exists()]
    partial = [name for name in expected if (model_dir / f"{name}.part").exists()]
    undersized = [name for name in expected if (model_dir / name).stat().st_size < 1_000_000_000]
    if missing or partial or undersized:
        raise RuntimeError(
            f"local Nemotron checkpoint is incomplete; missing={missing}, "
            f"partial={partial}, undersized={undersized}"
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
        loss = loss_for_batch(model, inputs, labels)
        count = labels.numel()
        total_loss += float(loss.detach().cpu()) * count
        total_tokens += count
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
        skip_svd_initialization=bool(
            args.init_cache
            and option_index in (1, 3)
            and Path(args.init_cache).exists()
        ),
    )
    if args.init_cache and option_index in (1, 3) and Path(args.init_cache).exists():
        cached_state = torch.load(args.init_cache, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(cached_state)
        # Older interrupted attempts may have been cached before the opening
        # gate fix; normalize that one scalar without altering other factors.
        for layer in model.layers:
            layer.attention.compressed_gate.data.fill_(-10.0)
        del cached_state
        print(f"reused SVD initialization for {option['name']} from {args.init_cache}", flush=True)
    elif args.init_cache and option_index == 0 and not Path(args.init_cache).exists():
        Path(args.init_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.trainable_state_dict(), args.init_cache)
        print(f"saved reusable SVD initialization to {args.init_cache}", flush=True)
    model.enable_gradient_checkpointing(args.gradient_checkpointing)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    corpus = CompositionLockedCorpus(spec, config.vocab_size, tokenizer=tokenizer)
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    candidate_root = checkpoint_root / f"{option_index:02d}_{option['name']}"
    manager = CheckpointManager(
        candidate_root,
        interval_steps=args.checkpoint_interval,
        max_checkpoints=args.max_checkpoints,
    )
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
            loss = loss_for_batch(model, inputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {option['name']} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            count = labels.numel()
            train_loss_sum += float(loss.detach().cpu()) * count
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
    # The optimizer is essential for preemption during a candidate, but not
    # for a completed architecture comparison.  Keep the model/RNG/metrics
    # state for continuation and reclaim the large Muon momentum snapshot so
    # four candidate finals fit on the local Windows volumes.
    optimizer_snapshot = final_checkpoint / "optimizer.pt"
    if optimizer_snapshot.exists():
        optimizer_snapshot.unlink()
    if args.final_archive_root:
        archive_root = Path(args.final_archive_root)
        if not archive_root.is_absolute():
            archive_root = ROOT / archive_root
        archive_path = archive_root / f"{option_index:02d}_{option['name']}"
        archive_root.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            raise FileExistsError(f"refusing to overwrite archived candidate {archive_path}")
        shutil.move(str(final_checkpoint), str(archive_path))
        final_checkpoint = archive_path
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
        "gpu_hours": train_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "estimated_cost_usd": (train_seconds / 3600.0) * args.local_gpu_hourly_cost
        if device.type == "cuda"
        else 0.0,
        "peak_vram_gib": peak_vram,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.parameter_count(trainable_only=True),
        "muon_matrix_parameters": getattr(optimizer, "matrix_parameter_count", None),
        "minimal_adamw_vector_parameters": getattr(optimizer, "vector_parameter_count", None),
        "checkpoint": str(final_checkpoint),
        "optimizer_snapshot_retained": False,
        "load_info": load_info.__dict__,
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=None)
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
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=1,
        help="retain only the newest atomic checkpoint per candidate on the local disk",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-root", default="checkpoints/real_ablations")
    parser.add_argument(
        "--final-archive-root",
        default=None,
        help="optional second-volume archive for stripped model-only completed candidates",
    )
    parser.add_argument(
        "--init-cache",
        default=None,
        help="optional CPU state cache; candidates 2 and 4 reuse candidate 1 SVD factors",
    )
    parser.add_argument("--output", default="reports/ablations/real_ablation_results.json")
    parser.add_argument(
        "--local-gpu-hourly-cost",
        type=float,
        default=0.0,
        help="Optional accounting rate; local ablation defaults to zero cash cost.",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.model_id is None:
        local_model = ROOT / "data/raw/nemotron"
        args.model_id = str(local_model) if (local_model / "model.safetensors.index.json").exists() else "nvidia/OpenReasoning-Nemotron-7B"
    validate_local_model_files(args.model_id)
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
        args.model_id,
        revision=args.revision,
        use_fast=False,
        local_files_only=Path(args.model_id).is_dir(),
    )
    results: list[dict[str, object]] = []
    for index, option in enumerate(OPTIONS):
        print(f"[{index + 1}/{len(OPTIONS)}] {option['name']}: {option}", flush=True)
        result = run_option(args, index, option, spec, tokenizer, device)
        results.append(result)
        retain_only_best_archive(results, args.final_archive_root)
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
