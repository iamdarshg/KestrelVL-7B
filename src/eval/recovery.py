"""Teacher-relative causal recovery metrics.

The metrics are deliberately reported separately.  NLL/KL and representation
similarity are diagnostics, not a fabricated benchmark-equivalent "retention"
number.  The legacy 0.95 capability gate can only be populated by a separate
held-out capability evaluation.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _output_value(output: object, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if not torch.is_tensor(value):
        raise RuntimeError(f"model output does not contain tensor {name!r}")
    return value


def _hidden_states(output: object) -> tuple[torch.Tensor, ...]:
    value = getattr(output, "hidden_states", None)
    if not isinstance(value, (tuple, list)) or not value:
        raise RuntimeError("teacher-recovery evaluation requires output_hidden_states=True")
    return tuple(item for item in value if torch.is_tensor(item))


def _causal_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [B,T,V] and labels must be [B,T]")
    scores = logits[:, :-1].float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("recovery labels contain no valid next-token targets")
    loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1), reduction="sum", ignore_index=-100)
    return loss / count, count


def _mean_token_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    teacher = teacher_logits[:, :-1].float()
    student = student_logits[:, :-1].float()
    valid = labels[:, 1:].ne(-100)
    denom = max(1, int(valid.sum().item()))
    teacher_logp = F.log_softmax(teacher, dim=-1)
    student_logp = F.log_softmax(student, dim=-1)
    teacher_p = teacher_logp.exp()
    forward = F.kl_div(student_logp, teacher_p, reduction="none").sum(-1)
    reverse = F.kl_div(teacher_logp, student_logp.exp(), reduction="none").sum(-1)
    return float(forward.masked_select(valid).sum().item() / denom), float(reverse.masked_select(valid).sum().item() / denom)


def _hidden_metrics(teacher: torch.Tensor, student: torch.Tensor) -> tuple[float, float]:
    if teacher.shape != student.shape:
        raise ValueError("teacher and candidate hidden states have different shapes")
    teacher_f = teacher.float()
    student_f = student.float()
    cosine = F.cosine_similarity(teacher_f, student_f, dim=-1).mean()
    norm_error = (student_f - teacher_f).square().mean().sqrt() / teacher_f.square().mean().sqrt().clamp_min(1e-12)
    return float(cosine.item()), float(norm_error.item())


def _mhc_diagnostics(model: torch.nn.Module) -> dict[str, float | int]:
    matrices = []
    for module in model.modules():
        if module.__class__.__name__ != "ManifoldHyperConnection" or not bool(getattr(module, "enabled", False)):
            continue
        matrix = module.matrix().float()
        matrices.append(matrix)
    if not matrices:
        return {"modules": 0, "max_row_sum_error": 0.0, "max_column_sum_error": 0.0, "max_nonfinite": 0}
    return {
        "modules": len(matrices),
        "max_row_sum_error": max(float((matrix.sum(-1) - 1).abs().max().item()) for matrix in matrices),
        "max_column_sum_error": max(float((matrix.sum(-2) - 1).abs().max().item()) for matrix in matrices),
        "max_nonfinite": int(any(not bool(torch.isfinite(matrix).all()) for matrix in matrices)),
    }


@torch.no_grad()
def evaluate_teacher_recovery(
    teacher: torch.nn.Module,
    candidate: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    selected_layers: Iterable[int] = (0, -1),
) -> dict[str, object]:
    """Evaluate one identical causal batch against a frozen teacher.

    ``input_ids`` is never shuffled or transformed here.  Callers must provide
    the frozen recovery split produced by :mod:`data.real_corpus`.
    """
    labels = input_ids if labels is None else labels
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must both have shape [B,T]")
    teacher.eval()
    candidate.eval()
    device = next(candidate.parameters()).device
    ids = input_ids.to(device)
    target = labels.to(device)
    teacher_out = teacher(ids, output_hidden_states=True)
    candidate_out = candidate(ids, output_hidden_states=True)
    teacher_logits = _output_value(teacher_out, "logits").to(device)
    candidate_logits = _output_value(candidate_out, "logits").to(device)
    teacher_nll, count = _causal_nll(teacher_logits, target)
    candidate_nll, _ = _causal_nll(candidate_logits, target)
    forward_kl, reverse_kl = _mean_token_kl(teacher_logits, candidate_logits, target)
    teacher_hidden = _hidden_states(teacher_out)
    candidate_hidden = _hidden_states(candidate_out)
    hidden: dict[str, dict[str, float]] = {}
    for layer in selected_layers:
        index = layer if layer >= 0 else len(teacher_hidden) + layer
        if not 0 <= index < len(teacher_hidden) or index >= len(candidate_hidden):
            raise ValueError(f"selected hidden-state layer {layer} is unavailable")
        cosine, error = _hidden_metrics(teacher_hidden[index], candidate_hidden[index].to(teacher_hidden[index].device))
        hidden[str(layer)] = {"cosine": cosine, "normalized_reconstruction_error": error}
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in (teacher_logits, candidate_logits))
    finite = finite and all(bool(torch.isfinite(tensor).all()) for tensor in teacher_hidden + candidate_hidden)
    return {
        "token_count": count,
        "teacher_nll": float(teacher_nll.item()),
        "candidate_nll": float(candidate_nll.item()),
        "delta_nll": float((candidate_nll - teacher_nll).item()),
        "forward_kl_teacher_to_candidate": forward_kl,
        "reverse_kl_candidate_to_teacher": reverse_kl,
        "hidden_state": hidden,
        "mhc": _mhc_diagnostics(candidate),
        "finite": finite,
        "teacher_capability_retention": None,
        "gate_status": "not_evaluable_without_held_out_capability_scores",
    }
