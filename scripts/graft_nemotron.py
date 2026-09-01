"""Initialize the real Nemotron -> Kestrel graft.

This entry point performs the architecture surgery and writes a resumable
stage checkpoint.  It deliberately does not pretend that initialization is
attention recovery: the new CSA/HCA/mHC path must still pass the measured
teacher-retention gate before CPT, SFT, or RL is allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.configuration import KestrelConfig
from model.nemotron import load_real_nemotron_transplant
from training.checkpoint import SafeCheckpointManager


def _default_model_id() -> str:
    local = ROOT / "data/raw/nemotron"
    return str(local) if (local / "model.safetensors.index.json").exists() else "nvidia/OpenReasoning-Nemotron-7B"


def _default_vision_id() -> str | None:
    # Prefer a complete local artifact.  A config-only or interrupted download
    # must not make the default command fail later during model construction.
    candidates = (
        ROOT / "data/raw/internvit",
        ROOT / "data/raw/tipsv2-l14",
    )
    for local in candidates:
        if (local / "config.json").exists() and (
            (local / "model.safetensors").exists()
            or (local / "model.safetensors.index.json").exists()
        ):
            return str(local)
    return None


def _pixels(path: str) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the real Nemotron Kestrel graft")
    parser.add_argument("--model-id", default=_default_model_id())
    parser.add_argument("--revision")
    parser.add_argument("--vision-model-id", default=_default_vision_id())
    parser.add_argument("--without-vision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vision-stage", choices=["none", "projector", "last4", "upper12", "all"], default="projector")
    parser.add_argument("--stage", choices=["attention_initialized", "vision_projector", "vision_adapted"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trainable-layer-start", type=int, default=0)
    parser.add_argument("--init-cache", help="optional compatible trainable SVD state cache")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--resume")
    parser.add_argument("--smoke-prompt")
    parser.add_argument("--smoke-image")
    parser.add_argument(
        "--smoke-random-image",
        action="store_true",
        help="use one deterministic-shape random 448px RGB tile for integration smoke",
    )
    parser.add_argument(
        "--smoke-cpu-lm-head",
        action="store_true",
        help="smoke-only: keep the large LM head on CPU to avoid NF4 temp-workspace OOM",
    )
    parser.add_argument("--smoke-image-kind", default="ide")
    parser.add_argument("--smoke-vision-budget", type=int)
    parser.add_argument("--debug-finite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.smoke_image and args.without_vision:
        raise SystemExit("--smoke-image requires vision; remove --without-vision")
    if args.smoke_image and args.smoke_random_image:
        raise SystemExit("choose --smoke-image or --smoke-random-image, not both")
    if args.smoke_random_image and args.without_vision:
        raise SystemExit("--smoke-random-image requires vision; remove --without-vision")
    if args.smoke_cpu_lm_head and not args.smoke_prompt:
        raise SystemExit("--smoke-cpu-lm-head is only valid with --smoke-prompt")
    use_vision = not args.without_vision
    if use_vision and not args.vision_model_id:
        raise SystemExit(
            "No complete local vision checkpoint is present. Run the InternViT "
            "or TIPSv2 downloader, pass --vision-model-id, or use --without-vision "
            "for the attention-only graft."
        )
    if args.stage is None:
        args.stage = "vision_projector" if use_vision else "attention_initialized"
    if args.stage == "attention_initialized" and use_vision:
        raise SystemExit("attention_initialized is text-only; use --stage vision_projector for the multimodal graft")
    if args.stage != "attention_initialized" and not use_vision:
        raise SystemExit(f"{args.stage} requires InternViT; omit --without-vision")
    if args.stage == "vision_projector" and args.vision_stage != "projector":
        raise SystemExit("vision_projector must use --vision-stage projector")
    if args.stage == "vision_adapted" and args.vision_stage == "none":
        raise SystemExit("vision_adapted requires a trainable vision stage")

    device = torch.device(args.device)
    if args.load_in_4bit and device.type != "cuda":
        raise SystemExit("real Nemotron Q4 grafting requires CUDA; use the tiny model for CPU-only tests")
    compute_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    config = KestrelConfig(use_vision=use_vision)
    model, load_info = load_real_nemotron_transplant(
        config,
        model_id=args.model_id,
        revision=args.revision,
        device=device,
        load_in_4bit=args.load_in_4bit,
        compute_dtype=compute_dtype,
        skip_svd_initialization=bool(args.init_cache),
        vision_model_id=args.vision_model_id if use_vision else None,
        vision_stage=args.vision_stage if use_vision else "none",
        cpu_lm_head=args.smoke_cpu_lm_head,
    )
    if args.init_cache:
        state_path = Path(args.init_cache)
        if not state_path.exists():
            raise SystemExit(f"initialization cache does not exist: {state_path}")
        model.load_trainable_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    model.freeze_backbone(args.trainable_layer_start)
    if use_vision:
        model.set_vision_trainable(args.vision_stage)
    model.debug_finite = args.debug_finite

    if args.checkpoint_root:
        checkpoint_root = Path(args.checkpoint_root)
    else:
        checkpoint_root = ROOT / "checkpoints" / ("04_vision_projector" if use_vision else "01_attention_initialized")
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    manager = SafeCheckpointManager(
        checkpoint_root,
        interval_steps=1,
        max_checkpoints=1,
        checkpoint_metadata={
            "config": config.to_dict(),
            "model_id": args.model_id,
            "revision": args.revision,
            "vision_model_id": args.vision_model_id if use_vision else None,
            "vision_stage": args.vision_stage if use_vision else "none",
            "cpu_lm_head": args.smoke_cpu_lm_head,
            "stage": args.stage,
            "protocol": {
                "kind": "real_nemotron_graft_initialization",
                "load_in_4bit": args.load_in_4bit,
                "compute_dtype": str(compute_dtype),
                "trainable_layer_start": args.trainable_layer_start,
            },
        },
    )
    step = 0
    if args.resume:
        state = manager.load(args.resume, model)
        step = int(state["step"])

    smoke: dict[str, object] = {}
    if args.smoke_prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            revision=args.revision,
            use_fast=False,
            local_files_only=Path(args.model_id).is_dir(),
        )
        ids = tokenizer(args.smoke_prompt, return_tensors="pt").input_ids.to(device)
        pixels = _pixels(args.smoke_image).to(device) if args.smoke_image else None
        if args.smoke_random_image:
            pixels = torch.rand(3, 448, 448, device=device, dtype=compute_dtype)
        model.eval()
        with torch.inference_mode():
            output = model(
                ids,
                pixel_values=pixels,
                vision_kind=args.smoke_image_kind,
                vision_budget=args.smoke_vision_budget,
                logits_to_keep=1,
            )
        smoke = {
            "prompt_tokens": int(ids.shape[1]),
            "logit_shape": list(output.logits.shape),
            "vision_telemetry": model.last_vision_telemetry if pixels is not None else None,
            "finite": bool(torch.isfinite(output.logits).all()),
        }
        if not smoke["finite"]:
            raise FloatingPointError("non-finite logits during real Nemotron graft smoke")

    checkpoint = manager.save(
        step,
        model,
        dataset_state={
            "token_cursor": 0,
            "corpus_fingerprint": None,
            "note": "initialization only; attention recovery has not run",
        },
        metrics={
            "status": "initialized_graft",
            "stage": args.stage,
            "load_info": load_info.__dict__,
            "model_parameter_count": model.parameter_count(),
            "trainable_parameter_count": model.parameter_count(trainable_only=True),
            "smoke": smoke,
        },
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "checkpoint": str(checkpoint),
                "load_info": load_info.__dict__,
                "config": config.to_dict(),
                "smoke": smoke,
                "note": "graft initialized; complete attention/hidden-state recovery before post-training",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
