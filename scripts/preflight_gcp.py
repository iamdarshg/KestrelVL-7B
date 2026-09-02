"""Run the mandatory local gates before any GCP benchmark is authorized.

This command is local-only.  It never invokes gcloud, torchrun, SSH, or a
remote process.  The generated JSON is consumed by the GCP launcher as an
explicit proof that the local validation gates were completed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "passed": result.returncode == 0,
    }
    if result.stdout:
        record["stdout_tail"] = result.stdout[-4000:]
    if result.stderr:
        record["stderr_tail"] = result.stderr[-4000:]
    print(f"[{ 'PASS' if record['passed'] else 'FAIL' }] {name} ({elapsed:.1f}s)", flush=True)
    return record


def _artifact_gate() -> dict[str, Any]:
    required = {
        "nemotron_index": ROOT / "data/raw/nemotron/model.safetensors.index.json",
        "tipsv2_config": ROOT / "data/raw/tipsv2-l14/config.json",
        "tipsv2_weights": ROOT / "data/raw/tipsv2-l14/model.safetensors",
        "tipsv2_manifest": ROOT / "data/raw/tipsv2-l14/download_manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    return {"name": "local_artifacts", "passed": not missing, "missing": missing}


def _graft_gate(path: Path) -> dict[str, Any]:
    latest = path / "latest"
    if not latest.is_file():
        return {"name": "real_multimodal_graft_smoke", "passed": False, "reason": "latest pointer missing"}
    checkpoint = path / latest.read_text(encoding="utf-8").strip()
    state_path = checkpoint / "state.json"
    if not state_path.is_file():
        return {"name": "real_multimodal_graft_smoke", "passed": False, "reason": "state.json missing"}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    smoke = state.get("metrics", {}).get("smoke", {})
    passed = bool(smoke.get("finite")) and bool(smoke.get("vision_telemetry"))
    return {
        "name": "real_multimodal_graft_smoke",
        "passed": passed,
        "checkpoint": str(checkpoint),
        "smoke": smoke,
        "reason": None if passed else "checkpoint does not contain finite multimodal smoke evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/gcp/local_preflight.json")
    parser.add_argument("--skip-vision-cuda", action="store_true")
    parser.add_argument("--max-rss-gib", type=float, default=1.3)
    args = parser.parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    gates = [_artifact_gate()]
    commands = [
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"]),
        (
            "ruff_new_paths",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "scripts/benchmark_gcp_throughput.py",
                "src/model/vision/internvit.py",
                "tests/test_attention.py",
                "--output-format",
                "concise",
            ],
        ),
        (
            "single_l4_profile_dry_run",
            [
                sys.executable,
                "scripts/benchmark_gcp_throughput.py",
                "--profile",
                "configs/hardware/gcp_single_l4.yaml",
                "--dry-run",
            ],
        ),
        (
            "final_rtx_pro_6000_profile_dry_run",
            [
                sys.executable,
                "scripts/benchmark_gcp_throughput.py",
                "--profile",
                "configs/hardware/gcp_final_rtx_pro_6000.yaml",
                "--dry-run",
            ],
        ),
    ]
    if not args.skip_vision_cuda:
        commands.append(
            (
                "tipsv2_cuda_smoke",
                [
                    sys.executable,
                    "scripts/verify_vision_checkpoint.py",
                    "data/raw/tipsv2-l14",
                    "--device",
                    "cuda",
                    "--max-tiles",
                    "1",
                    "--max-rss-gib",
                    str(args.max_rss_gib),
                ],
            )
        )
    for name, command in commands:
        gates.append(_run(name, command))
    gates.append(_graft_gate(ROOT / "checkpoints/04_vision_projector_tipsv2_smoke"))
    passed = all(bool(gate.get("passed")) for gate in gates)
    report = {
        "status": "pass" if passed else "fail",
        "local_only": True,
        "generated_at": time.time(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gates": gates,
        "gcp_authorized": False,
        "note": "A human must still authorize the GCP command and budget after reviewing this report.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output_path), "gcp_authorized": False}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
