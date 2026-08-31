import json
import time
from pathlib import Path

import torch


def snapshot(device: torch.device | str = "cuda") -> dict[str, object]:
    result: dict[str, object] = {"time": time.time(), "device": str(device), "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        index = torch.device(device).index or torch.cuda.current_device()
        result.update({"name": torch.cuda.get_device_name(index), "allocated_bytes": torch.cuda.memory_allocated(index), "reserved_bytes": torch.cuda.memory_reserved(index), "peak_bytes": torch.cuda.max_memory_allocated(index), "total_bytes": torch.cuda.get_device_properties(index).total_memory})
    return result


def assert_vram_budget(max_gib: float, device: torch.device | str = "cuda") -> dict[str, object]:
    result = snapshot(device)
    peak = float(result.get("peak_bytes", 0)) / 2**30
    if torch.cuda.is_available() and peak > max_gib:
        raise MemoryError(f"VRAM budget exceeded: {peak:.3f} GiB > {max_gib:.3f} GiB")
    return result


def append_jsonl(path: str | Path, record: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

