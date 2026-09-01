"""Layerwise recovery loop with progressive unfreezing hooks."""

from collections.abc import Iterable

import torch

from .attention_distill import hidden_state_distillation_loss


def _last_hidden(output: object) -> torch.Tensor:
    hidden = getattr(output, "hidden_states", None)
    if isinstance(hidden, (tuple, list)):
        if not hidden:
            raise RuntimeError("model returned an empty hidden-state collection")
        return hidden[-1]
    if torch.is_tensor(hidden):
        return hidden
    logits = getattr(output, "logits", None)
    if torch.is_tensor(logits):
        return logits
    raise RuntimeError("model output has neither hidden_states nor logits")


def _logits(output: object, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if torch.is_tensor(logits) and logits.shape[-1] > 0:
        return logits
    head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("model output omitted logits and model has no lm_head")
    return head(hidden)


def recovery_step(student, teacher, input_ids: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    student_out = student(input_ids, output_hidden_states=True)
    with torch.no_grad():
        teacher_out = teacher(input_ids, output_hidden_states=True)
    student_hidden = _last_hidden(student_out)
    teacher_hidden = _last_hidden(teacher_out).to(student_hidden.device)
    student_logits = _logits(student_out, student_hidden, student)
    teacher_logits = _logits(teacher_out, teacher_hidden, teacher).to(student_logits.device)
    loss = hidden_state_distillation_loss(
        student_logits,
        teacher_logits,
        student_hidden,
        teacher_hidden,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def set_trainable_subsets(model: torch.nn.Module, names: Iterable[str]) -> None:
    names = tuple(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(token in name for token in names))
