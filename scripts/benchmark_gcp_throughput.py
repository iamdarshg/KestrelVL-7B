"""Benchmark the real Nemotron graft on a GCP GPU profile.

The harness deliberately measures the architecture that will be trained rather
than the tiny unit-test model.  It uses a synthetic token stream so the result
is a loader/kernel/optimizer measurement, not a data-quality claim.  For the
dual-L4 profile each process owns one complete replica and DDP includes the
gradient synchronization cost.  Per-device microbatch is always one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.configuration import KestrelConfig  # noqa: E402
from model.nemotron import load_real_nemotron_transplant  # noqa: E402
from training.muon import build_muon_optimizer  # noqa: E402


def _load_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise ValueError(f"hardware profile must be a mapping: {path}")
    if int(profile.get("per_device_micro_batch_size", 0)) != 1:
        raise ValueError("the GCP harness requires per_device_micro_batch_size=1")
    return profile


def _profile_value(profile: dict[str, Any], key: str, default: Any) -> Any:
    value = profile.get(key, default)
    return default if value is None else value


def _distributed_context() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun data parallel requires CUDA")
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size, distributed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _telemetry(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device)}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "current_allocated_bytes": torch.cuda.memory_allocated(device),
        "current_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _write_record(path: Path, record: dict[str, Any], rank: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


def _dry_run(profile_path: Path, profile: dict[str, Any], args: argparse.Namespace) -> None:
    distributed = str(profile.get("distributed", "single_gpu"))
    requested_world = 2 if distributed == "data_parallel" else 1
    output = {
        "profile": str(profile_path),
        "name": profile.get("name"),
        "machine_type": profile.get("machine_type"),
        "accelerator": profile.get("accelerator"),
        "accelerator_count": profile.get("accelerator_count"),
        "distributed": distributed,
        "expected_world_size": requested_world,
        "per_device_micro_batch_size": profile.get("per_device_micro_batch_size"),
        "sequence_length": args.sequence_length
        or int(_profile_value(profile, "sequence_length", 8192)),
        "gradient_accumulation_steps": args.gradient_accumulation_steps
        or int(_profile_value(profile, "gradient_accumulation_steps", 1)),
        "mode": args.mode,
        "model_id": args.model_id,
        "vision_model_id": args.vision_model_id,
        "load_in_4bit": not args.no_4bit,
        "note": "dry run only; no model loaded and no throughput claim",
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/hardware/gcp_dual_l4.yaml")
    parser.add_argument("--model-id", default="data/raw/nemotron")
    parser.add_argument("--vision-model-id")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--mode", choices=("train", "forward"), default="train")
    parser.add_argument("--output", default="reports/runtime/gcp_throughput.json")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = _load_profile(profile_path)
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup-steps must be non-negative")
    sequence_length = args.sequence_length or int(_profile_value(profile, "sequence_length", 8192))
    accumulation = args.gradient_accumulation_steps or int(
        _profile_value(profile, "gradient_accumulation_steps", 1)
    )
    if sequence_length < 1 or accumulation < 1:
        raise ValueError("sequence length and gradient accumulation must be positive")
    if args.dry_run:
        _dry_run(profile_path, profile, args)
        return

    expected_world_size = (
        int(profile.get("accelerator_count", 1))
        if profile.get("distributed") == "data_parallel"
        else 1
    )
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world_size != expected_world_size:
        raise ValueError(
            f"profile {profile.get('name', profile_path)} requires world size "
            f"{expected_world_size}, got {requested_world_size}"
        )
    rank, local_rank, world_size, distributed = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("the GCP throughput harness requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(device) else torch.float16

    use_vision = args.vision_model_id is not None
    config = KestrelConfig(
        use_vision=use_vision,
        max_position_embeddings=max(1_048_576, sequence_length),
    )
    model, load_info = load_real_nemotron_transplant(
        config,
        model_id=args.model_id,
        device=device,
        load_in_4bit=not args.no_4bit,
        compute_dtype=compute_dtype,
        vision_model_id=args.vision_model_id,
        vision_stage="projector" if use_vision else "none",
    )
    model.enable_gradient_checkpointing(bool(_profile_value(profile, "gradient_checkpointing", True)))
    model.train(args.mode == "train")
    optimizer = None
    if args.mode == "train":
        optimizer = build_muon_optimizer(model, weight_decay=0.0)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    tokens_per_micro_step = sequence_length * world_size
    pixels = None
    if use_vision:
        pixels = torch.rand(3, 448, 448, device=device, dtype=compute_dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    total_steps = args.warmup_steps + args.steps
    measured_start = None
    measured_micro_steps = 0
    measured_loss = 0.0
    start = time.perf_counter()
    for step in range(total_steps):
        if args.mode == "train" and step % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        ids = torch.randint(
            0,
            config.vocab_size,
            (1, sequence_length),
            device=device,
        )
        sync_context = (
            model.no_sync()  # type: ignore[union-attr]
            if distributed and args.mode == "train" and (step + 1) % accumulation != 0
            else nullcontext()
        )
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                output = model(
                    ids,
                    pixel_values=pixels,
                    vision_kind="ide",
                    output_hidden_states=True,
                )
                if output.hidden_states is None:
                    raise RuntimeError("real model did not return hidden states")
                loss = output.hidden_states.float().square().mean()
            if args.mode == "train":
                (loss / accumulation).backward()
        if args.mode == "train" and (step + 1) % accumulation == 0:
            optimizer.step()  # type: ignore[union-attr]
        if step + 1 == args.warmup_steps:
            _synchronize(device)
            measured_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        if step >= args.warmup_steps:
            measured_micro_steps += 1
            measured_loss += float(loss.detach().cpu())
    _synchronize(device)
    elapsed = time.perf_counter() - (measured_start or start)
    if args.warmup_steps == 0:
        measured_micro_steps = args.steps
    tokens = measured_micro_steps * tokens_per_micro_step
    hourly_rate = float(_profile_value(profile, "estimated_spot_usd_per_node_hour", 0.0))
    record = {
        "profile": str(profile_path),
        "profile_name": profile.get("name"),
        "model_id": args.model_id,
        "vision_model_id": args.vision_model_id,
        "mode": args.mode,
        "sequence_length": sequence_length,
        "per_device_micro_batch_size": 1,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "warmup_steps": args.warmup_steps,
        "measured_micro_steps": measured_micro_steps,
        "tokens_processed": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "micro_steps_per_second": measured_micro_steps / max(elapsed, 1e-9),
        "mean_proxy_loss": measured_loss / max(measured_micro_steps, 1),
        "gpu_hours": elapsed * world_size / 3600.0,
        "estimated_cost_usd": elapsed / 3600.0 * hourly_rate,
        "load_info": load_info.__dict__,
        "telemetry": _telemetry(device),
        "synthetic_workload": True,
        "lm_head_included": False,
        "note": "architecture throughput only; output_hidden_states uses a bounded proxy loss and skips the 152k-vocabulary LM head",
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    _write_record(output_path, record, rank)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank == 0:
        print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
