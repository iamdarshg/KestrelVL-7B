"""Precision/optimizer policy validation and accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    forward_weight_storage: str = "q4"
    compute_dtype: torch.dtype = torch.bfloat16
    master_weight_dtype: torch.dtype = torch.bfloat16
    gradient_dtype: torch.dtype = torch.bfloat16
    allow_fp32_master: bool = False
    muon_momentum_dtype: torch.dtype = torch.bfloat16


def validate_precision_policy(model: torch.nn.Module, policy: PrecisionPolicy) -> dict[str, Any]:
    """Fail loudly on accidental full-FP32 trainable/master copies."""
    if policy.forward_weight_storage not in {"q4", "nf4"}:
        raise ValueError("forward weights must use a declared Q4/NF4 storage policy")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fp32_trainable = sum(parameter.numel() for parameter in trainable if parameter.dtype == torch.float32)
    if fp32_trainable and not policy.allow_fp32_master:
        raise TypeError(
            f"precision policy forbids FP32 master/trainable parameters; found {fp32_trainable} elements"
        )
    return {
        "forward_weight_storage": policy.forward_weight_storage,
        "compute_dtype": str(policy.compute_dtype),
        "master_weight_dtype": str(policy.master_weight_dtype),
        "gradient_dtype": str(policy.gradient_dtype),
        "allow_fp32_master": policy.allow_fp32_master,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "fp32_trainable_parameters": fp32_trainable,
    }


def optimizer_telemetry(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Report Muon/AdamW coverage and optimizer state bytes."""
    muon_params = 0
    adamw_params = 0
    for group in optimizer.param_groups:
        count = sum(parameter.numel() for parameter in group["params"])
        if group.get("use_muon", False):
            muon_params += count
        else:
            adamw_params += count
    state_bytes = 0
    state_tensors = 0
    for values in optimizer.state.values():
        for value in values.values():
            if torch.is_tensor(value):
                state_bytes += value.numel() * value.element_size()
                state_tensors += 1
    total = muon_params + adamw_params
    return {
        "muon_parameter_count": muon_params,
        "adamw_parameter_count": adamw_params,
        "adamw_fraction": adamw_params / max(1, total),
        "state_bytes": state_bytes,
        "state_tensors": state_tensors,
    }
