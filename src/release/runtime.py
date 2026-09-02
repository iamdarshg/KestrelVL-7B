"""Memory-bounded runtime for pickle-free Kestrel Q4 bundles.

The serializer stores ordinary linear weights as packed int4 nibbles.  A
naive loader that immediately dequantizes every tensor recreates a dense
BF16 model and defeats the RTX 4060 memory target.  ``Q4Linear`` keeps the
packed weight and its per-group scales as buffers; only the layer currently
executing is dequantized for the matrix multiply.

This is a correctness/reference runtime.  A fused CUDA dequantize-GEMM
kernel can replace its ``forward`` implementation later without changing the
release format or loader contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file

from .serialization import unpack_q4_tensor


class Q4Linear(nn.Module):
    """Linear layer whose weight remains packed until execution."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        metadata: dict[str, Any],
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if packed.dtype != torch.uint8:
            raise TypeError("Q4Linear packed weight must be uint8")
        shape = tuple(int(value) for value in metadata["shape"])
        if len(shape) != 2:
            raise ValueError(f"Q4Linear expects a 2-D weight, got {shape}")
        if scales.ndim != 1:
            raise ValueError("Q4Linear scales must be one-dimensional")
        self.register_buffer("packed_weight", packed.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.metadata = dict(metadata)
        self.in_features = shape[1]
        self.out_features = shape[0]
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_bundle(
        cls,
        tensors: dict[str, torch.Tensor],
        entry: dict[str, Any],
        bias: torch.Tensor | None = None,
    ) -> "Q4Linear":
        return cls(tensors[entry["packed"]], tensors[entry["scales"]], entry, bias=bias)

    def dequantize(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Materialize one layer's weight on the compute device."""
        return unpack_q4_tensor(self.packed_weight, self.scales, self.metadata).to(
            device=device, dtype=dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(input.dtype, input.device)
        bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias)


def _parent_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    if not parts:
        raise ValueError("empty module name")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _assign_tensor(root: nn.Module, name: str, value: torch.Tensor) -> None:
    """Assign a native bundle tensor to a parameter or buffer by state name."""
    module_name, leaf = name.rsplit(".", 1)
    parent = root
    for part in module_name.split("."):
        parent = getattr(parent, part)
    if leaf in parent._parameters:
        parameter = parent._parameters[leaf]
        if parameter is None or tuple(parameter.shape) != tuple(value.shape):
            parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        else:
            parameter.data.copy_(value.to(parameter.device, parameter.dtype))
        return
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
        return
    raise KeyError(f"bundle tensor {name!r} is not present in the model")


def load_q4_runtime(model: nn.Module, path: str | Path, device: str | torch.device = "cuda") -> nn.Module:
    """Load a Q4 bundle into ``model`` without global dequantization.

    The model must have the same module topology used to create the bundle.
    Ordinary linear weights are replaced with ``Q4Linear`` modules.  Native
    tensors (embeddings, norms, sensitive modules, biases) are copied in the
    requested device/dtype.  The function intentionally performs strict
    topology checks so a mismatched release cannot silently produce nonsense.
    """
    root = Path(path)
    manifest_path = root / "quantization_config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Q4 manifest not found: {manifest_path}")
    # Validate the artifact digest directly.  The runtime intentionally does
    # not call the debug ``load_q4_bundle`` path here because that path
    # dequantizes every tensor into a dense dictionary.  Packed tensors are
    # loaded once onto the target device and each linear weight is
    # dequantized only for its active matmul.
    checksum_path = root / "checksums.json"
    if checksum_path.exists():
        import hashlib

        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
        if checksums.get("model.safetensors") != observed:
            raise ValueError("model.safetensors checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pickle_free_loader") is not True:
        raise ValueError("release is missing the pickle-free loader marker")
    target = torch.device(device)
    tensors = load_file(str(root / "model.safetensors"), device=str(target))
    entries = manifest.get("weights", {})
    if not entries:
        raise ValueError("Q4 release contains no weights")

    # Replace quantized linear leaves first.  A bias entry is native and is
    # attached after replacement so the resulting module retains the normal
    # ``module.bias`` state name.
    quantized = {name: entry for name, entry in entries.items() if entry.get("storage") == "q4"}
    for weight_name, entry in quantized.items():
        if not weight_name.endswith(".weight"):
            # Tiny non-linear parameters (mHC logits, scalar gates, and
            # grouped projection tensors) are rare and cheap to materialize.
            # Keep the large matrix path packed, while still accepting every
            # tensor emitted by the generic serializer.
            _assign_tensor(
                model,
                weight_name,
                unpack_q4_tensor(
                    tensors[entry["packed"]], tensors[entry["scales"]], entry
                ).to(target),
            )
            continue
        module_name = weight_name[: -len(".weight")]
        parent, leaf = _parent_module(model, module_name)
        old = getattr(parent, leaf, None)
        if not isinstance(old, nn.Linear):
            raise TypeError(
                f"Q4 weight {weight_name} expects nn.Linear at {module_name}, got {type(old).__name__}"
            )
        bias_entry = entries.get(f"{module_name}.bias")
        bias = None
        if bias_entry is not None:
            if bias_entry.get("storage") != "native":
                raise ValueError(f"linear bias must be native: {module_name}.bias")
            bias = tensors[bias_entry["tensor"]]
        replacement = Q4Linear.from_bundle(tensors, entry, bias=bias)
        setattr(parent, leaf, replacement)

    # Load native state after replacements.  Quantized weight entries have no
    # original ``.weight`` parameter anymore and are deliberately skipped.
    for name, entry in entries.items():
        if entry.get("storage") != "native" or name.endswith(".bias") and name[:-5] in {
            weight_name[: -len(".weight")] for weight_name in quantized
        }:
            continue
        _assign_tensor(model, name, tensors[entry["tensor"]])

    # Every bundle key must have been consumed.  This catches a model/config
    # mismatch while the offending artifact is still easy to diagnose.
    known_native = {
        entry["tensor"] for entry in entries.values() if entry.get("storage") == "native"
    }
    known_quantized = {
        key
        for entry in entries.values()
        if entry.get("storage") == "q4"
        for key in (entry["packed"], entry["scales"])
    }
    unused = set(tensors) - known_native - known_quantized
    if unused:
        raise ValueError(f"release contains unreferenced tensors: {sorted(unused)[:3]}")
    model.to(target)
    return model


__all__ = ["Q4Linear", "load_q4_runtime"]
