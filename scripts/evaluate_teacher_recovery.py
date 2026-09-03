"""Run the teacher-relative recovery diagnostics on a fixed token batch.

The default tiny mode is a local instrumentation smoke.  Real Nemotron use
requires a caller-supplied token tensor and model-loading environment; this
script intentionally never substitutes synthetic data for a production gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.recovery import evaluate_teacher_recovery  # noqa: E402
from model import KestrelConfig, KestrelForCausalLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ids", help="torch-saved [B,T] tensor; omit for the tiny instrumentation smoke")
    parser.add_argument("--output", default="reports/recovery/teacher_recovery_smoke.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = KestrelConfig.tiny(use_vision=False)
    teacher = KestrelForCausalLM(config).to(device).eval()
    candidate = KestrelForCausalLM(config).to(device).eval()
    candidate.load_state_dict(teacher.state_dict())
    if args.input_ids:
        ids = torch.load(args.input_ids, map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator().manual_seed(20260831)
        ids = torch.randint(0, config.vocab_size, (1, 32), generator=generator)
    if not torch.is_tensor(ids):
        raise TypeError("--input-ids must contain a tensor")
    report = evaluate_teacher_recovery(teacher, candidate, ids, selected_layers=(0, -1))
    report["evidence_label"] = "local_tiny_identical_teacher_candidate_smoke"
    report["real_data_gate"] = False
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
