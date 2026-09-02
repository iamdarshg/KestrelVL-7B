"""Load and image-smoke a local InternViT or TIPSv2 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.vision.internvit import InternViTEncoder, dynamic_tiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image")
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--trainable-smoke", action="store_true")
    parser.add_argument("--max-rss-gib", type=float)
    args = parser.parse_args()

    if args.max_rss_gib is not None:
        # Keep CPU thread-pool arenas bounded on the small Windows host.  The
        # image tower is CUDA-resident; extra CPU workers only inflate RSS.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def check_rss(phase: str) -> float | None:
        if args.max_rss_gib is None:
            return None
        import psutil

        rss = psutil.Process().memory_info().rss / 2**30
        if rss > args.max_rss_gib:
            raise MemoryError(
                f"RSS budget exceeded during {phase}: {rss:.3f} GiB > {args.max_rss_gib:.3f} GiB"
            )
        return rss

    def trim_working_set() -> None:
        # The streaming loader touches each safetensors range once.  Windows
        # may retain those clean file-backed pages in the process working set
        # even after the tensors have been copied to CUDA.  Trim only clean
        # pages before the measured forward; this changes residency, not model
        # state, and keeps the RSS gate meaningful on the constrained host.
        if args.max_rss_gib is None or sys.platform != "win32":
            return
        import ctypes

        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetProcessWorkingSetSize(process, ctypes.c_size_t(-1), ctypes.c_size_t(-1))

    stop_watchdog = threading.Event()
    watchdog = None
    if args.max_rss_gib is not None:
        import psutil

        def watch_rss() -> None:
            # Poll below the public ceiling so the process cannot run through
            # the limit between two coarse phase checks.
            threshold = max(0.1, args.max_rss_gib - 0.02)
            while not stop_watchdog.wait(0.02):
                rss = psutil.Process().memory_info().rss / 2**30
                if rss > threshold:
                    print(
                        f"RSS watchdog stopping process: {rss:.3f} GiB > {threshold:.3f} GiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(86)

        watchdog = threading.Thread(target=watch_rss, name="rss-watchdog", daemon=True)
        watchdog.start()

    path = Path(args.model_path)
    if not (path / "config.json").exists():
        raise SystemExit(f"missing config.json: {path}")
    if not ((path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()):
        raise SystemExit(f"no complete model weight file found: {path}")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        encoder = InternViTEncoder(
            model_path=path,
            freeze=not args.trainable_smoke,
            torch_dtype=dtype,
            device=device,
        )
        load_rss_gib = check_rss("vision load")
        trim_working_set()
        if args.image:
            from PIL import Image
            import numpy as np

            image = Image.open(args.image).convert("RGB")
            pixels = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        else:
            pixels = torch.rand(3, 448, 448)
        pixels = dynamic_tiles(pixels, max_tiles=args.max_tiles)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        encoder.eval()
        trim_working_set()
        context = torch.enable_grad() if args.trainable_smoke else torch.inference_mode()
        with context:
            tokens = encoder.encode_with_policy(pixels.to(device), device, offload_to_cpu=False)
        result = {
            "path": str(path.resolve()),
            "backend": encoder.backend,
            "hidden_size": encoder.hidden_size,
            "tiles": int(pixels.shape[0]),
            "tokens": int(tokens.shape[1]),
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "finite": bool(torch.isfinite(tokens).all()),
            "telemetry": encoder.last_telemetry,
            "load_process_rss_gib": load_rss_gib,
        }
        if device.type == "cuda":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        result["process_rss_gib"] = check_rss("vision forward")
        if not result["finite"]:
            raise FloatingPointError("vision checkpoint emitted non-finite tokens")
        print(json.dumps(result, indent=2, default=str))
    finally:
        stop_watchdog.set()
        if watchdog is not None:
            watchdog.join(timeout=1.0)


if __name__ == "__main__":
    main()
