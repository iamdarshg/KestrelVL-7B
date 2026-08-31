"""Layerwise recovery loop with progressive unfreezing hooks."""

from collections.abc import Iterable

import torch

from .attention_distill import hidden_state_distillation_loss


def recovery_step(student, teacher, input_ids: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    student_out = student(input_ids)
    with torch.no_grad():
        teacher_out = teacher(input_ids)
    loss = hidden_state_distillation_loss(student_out.logits, teacher_out.logits, student_out.logits, teacher_out.logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def set_trainable_subsets(model: torch.nn.Module, names: Iterable[str]) -> None:
    names = tuple(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(token in name for token in names))

