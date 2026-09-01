"""Resumable, cost-neutral stage bootstrap and gate-aware launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from model import KestrelConfig, KestrelForCausalLM
from training.checkpoint import SafeCheckpointManager

STAGES = ["original_teacher", "attention_initialized", "attention_reconstructed", "hidden_state_recovered", "vision_projector", "vision_adapted", "cpt", "sft", "preference", "grpo_code", "grpo_swe", "grpo_terminal", "grpo_vision", "grpo_long", "opd", "final_grpo", "context_extended", "qat", "q4_release"]


def digest_files() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "references").glob("*.json")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", default="configs/hardware/local_4060_8gb.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--output-root", default="checkpoints")
    args = parser.parse_args()
    if args.stage in {"sft", "preference", "grpo_code", "grpo_swe", "grpo_terminal", "grpo_vision", "grpo_long", "opd", "final_grpo", "context_extended", "qat", "q4_release"}:
        gate = ROOT / "reports" / "reconstruction_gate.json"
        if not gate.exists():
            raise SystemExit(f"refusing {args.stage}: missing {gate}; complete measured recovery first")
    config = KestrelConfig.tiny(use_vision=False)
    model = KestrelForCausalLM(config)
    manager = SafeCheckpointManager(
        ROOT / args.output_root / f"{STAGES.index(args.stage):02d}_{args.stage}",
        interval_steps=25,
        checkpoint_metadata={"config": config.to_dict(), "stage": args.stage},
    )
    if args.resume:
        manager.load(args.resume, model)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(ids).logits
    target = manager.save(0, model, dataset_state={"cursor": 0, "manifest_sha256": digest_files()}, metrics={"stage": args.stage, "logit_shape": list(logits.shape), "status": "initialized_only"})
    print(json.dumps({"stage": args.stage, "checkpoint": str(target), "status": "initialized_only", "note": "no training is implied by a bootstrap checkpoint"}, indent=2))


if __name__ == "__main__":
    main()
