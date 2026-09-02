"""Q4 bundle serialization using safetensors only.

The training checkpoint format is intentionally separate from this release
format.  Release loading never invokes pickle or ``torch.load``.  Sensitive
small modules can remain BF16/FP16 while ordinary floating tensors use
per-group symmetric int4 packing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def pack_q4_tensor(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Pack a floating tensor into two int4 values per uint8 byte."""
    if not tensor.is_floating_point():
        raise TypeError("Q4 packing requires a floating tensor")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    original_shape = list(tensor.shape)
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    padded_length = ((flat.numel() + group_size - 1) // group_size) * group_size
    if padded_length == 0:
        padded_length = group_size
    padded = torch.zeros(padded_length, dtype=torch.float32)
    padded[: flat.numel()] = flat
    grouped = padded.view(-1, group_size)
    scales = grouped.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    quantized = (grouped / scales[:, None]).round().clamp(-8, 7).to(torch.int16) + 8
    quantized = quantized.to(torch.uint8).reshape(-1)
    if quantized.numel() % 2:
        quantized = torch.cat((quantized, torch.zeros(1, dtype=torch.uint8)))
    packed = quantized[0::2] | (quantized[1::2] << 4)
    metadata = {
        "shape": original_shape,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "group_size": group_size,
        "padded_numel": padded_length,
        "scheme": "symmetric_int4_uint8_nibbles_offset_8",
    }
    return packed, scales.to(torch.float16), metadata


def unpack_q4_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    metadata: Mapping[str, Any],
) -> torch.Tensor:
    """Dequantize a packed Q4 tensor according to its manifest entry."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed Q4 data must be uint8")
    low = packed & 0x0F
    high = packed >> 4
    values = torch.stack((low, high), dim=1).reshape(-1).to(torch.int16) - 8
    padded_numel = int(metadata["padded_numel"])
    values = values[:padded_numel].reshape(-1, int(metadata["group_size"]))
    values = values * scales.to(torch.float32)[:, None]
    values = values.reshape(-1)[: int(torch.tensor(metadata["shape"]).prod().item())]
    dtype = getattr(torch, str(metadata["dtype"]))
    return values.reshape(tuple(int(v) for v in metadata["shape"])).to(dtype)


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_q4_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a release from metadata without dequantizing its tensors.

    This is the clean-process export gate.  ``safe_open`` reads the
    safetensors header and individual tensor metadata, so validation does not
    create a dense model-sized CPU copy merely to prove that a packed release
    is structurally loadable.
    """
    root = Path(path)
    required = (
        "model.safetensors",
        "config.json",
        "quantization_config.json",
        "checksums.json",
        "kestrel_runtime.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Q4 bundle is missing required files: {missing}")
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        artifact = root / name
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"Q4 bundle checksum mismatch: {name}")
    manifest = json.loads((root / "quantization_config.json").read_text(encoding="utf-8"))
    if manifest.get("pickle_free_loader") is not True:
        raise ValueError("Q4 bundle is missing the pickle-free loader marker")
    entries = manifest.get("weights")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Q4 bundle contains no weight entries")
    with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}
    referenced: set[str] = set()
    for name, entry in entries.items():
        storage = entry.get("storage")
        if storage == "q4":
            packed_name, scale_name = entry.get("packed"), entry.get("scales")
            if packed_name not in keys or scale_name not in keys:
                raise ValueError(f"Q4 entry {name!r} references a missing packed tensor")
            if dtypes[packed_name] != "U8" or dtypes[scale_name] not in {"F16", "BF16", "F32"}:
                raise ValueError(f"Q4 entry {name!r} has invalid packed/scales dtypes")
            referenced.update((packed_name, scale_name))
        elif storage == "native":
            tensor_name = entry.get("tensor")
            if tensor_name not in keys:
                raise ValueError(f"native entry {name!r} references a missing tensor")
            referenced.add(tensor_name)
        else:
            raise ValueError(f"unsupported Q4 storage for {name!r}: {storage!r}")
    unused = keys - referenced
    if unused:
        raise ValueError(f"Q4 bundle contains unreferenced tensors: {sorted(unused)[:3]}")
    return {
        "valid": True,
        "format": manifest.get("format"),
        "tensor_entries": len(entries),
        "packed_tensor_count": sum(entry.get("storage") == "q4" for entry in entries.values()),
        "safetensors_keys": len(keys),
        "metadata_only": True,
        "total_model_bytes": (root / "model.safetensors").stat().st_size,
        "tensor_shapes": shapes,
    }


def save_q4_bundle(
    state_dict: Mapping[str, torch.Tensor],
    output: str | Path,
    config: Mapping[str, Any],
    group_size: int = 128,
    sensitive_patterns: tuple[str, ...] = (
        "embed_tokens",
        "lm_head",
        "norm",
        "projector",
        "vision",
        "sink",
        "indexer",
    ),
    force: bool = False,
) -> Path:
    """Write an atomic, self-describing Q4 release directory."""
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"release directory exists; pass force=True to replace: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {}
    entries: dict[str, dict[str, Any]] = {}
    for name, tensor in state_dict.items():
        safe = _safe_name(name)
        if tensor.is_floating_point() and not any(pattern in name for pattern in sensitive_patterns):
            packed, scales, metadata = pack_q4_tensor(tensor, group_size)
            tensors[f"q4__{safe}"] = packed
            tensors[f"scale__{safe}"] = scales
            entries[name] = {"storage": "q4", "packed": f"q4__{safe}", "scales": f"scale__{safe}", **metadata}
        else:
            tensors[f"fp__{safe}"] = tensor.detach().to(device="cpu").contiguous()
            entries[name] = {
                "storage": "native",
                "tensor": f"fp__{safe}",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
    model_path = temporary / "model.safetensors"
    save_file(tensors, str(model_path), metadata={"format": "kestrel-q4-v1", "created_unix": str(time.time())})
    manifest = {
        "format": "kestrel-q4-v1",
        "quantization": "q4_symmetric_groupwise",
        "group_size": group_size,
        "weights": entries,
        "config": dict(config),
        "pickle_free_loader": True,
    }
    (temporary / "config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "quantization_config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "kestrel_runtime.json").write_text(
        json.dumps(
            {
                "format": "kestrel-q4-v1",
                "batch_size": 1,
                "ordinary_weights_gpu_resident": True,
                "compressed_state_cpu_offload_allowed": True,
                "loader": "safetensors_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = {
        file.name: _sha256_file(file)
        for file in sorted(temporary.iterdir())
        if file.is_file() and file.name != "checksums.json"
    }
    (temporary / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    os.replace(temporary, output)
    return output


def load_q4_bundle(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load and dequantize a release without executing pickle."""
    root = Path(path)
    checksum_path = root / "checksums.json"
    if checksum_path.exists():
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
        if checksums.get("model.safetensors") != observed:
            raise ValueError("model.safetensors checksum mismatch")
    manifest = json.loads((root / "quantization_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(root / "model.safetensors"), device=str(device))
    result: dict[str, torch.Tensor] = {}
    for name, entry in manifest["weights"].items():
        if entry["storage"] == "q4":
            result[name] = unpack_q4_tensor(
                tensors[entry["packed"]], tensors[entry["scales"]], entry
            ).to(device)
        else:
            result[name] = tensors[entry["tensor"]]
    return result
