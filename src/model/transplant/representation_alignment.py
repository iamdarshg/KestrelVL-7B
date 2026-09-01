"""Stable multi-signal teacher/student representation losses."""

import torch
import torch.nn.functional as F


def representation_loss(student: torch.Tensor, teacher: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"shape mismatch: {student.shape} != {teacher.shape}")
    if kind == "mse":
        return F.mse_loss(student.float(), teacher.float())
    if kind == "cosine":
        return (1 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()
    if kind == "normalized":
        sn = F.normalize(student.float(), dim=-1)
        tn = F.normalize(teacher.float(), dim=-1)
        return F.mse_loss(sn, tn)
    raise ValueError(f"unknown representation loss {kind}")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    direction: str = "forward",
) -> torch.Tensor:
    """Token-normalized KL divergence for causal recovery.

    ``reduction='batchmean'`` is correct for a single distribution per batch,
    but it scales a sequence-shaped tensor by the batch size only.  Recovery
    batches have thousands of token distributions, so that reduction makes
    the KL weight depend on sequence length.  Sum over vocabulary and average
    over every remaining distribution instead.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"shape mismatch: {student_logits.shape} != {teacher_logits.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    if direction == "forward":
        values = F.kl_div(
            F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none"
        ).sum(dim=-1)
        return values.reshape(-1).mean() * temperature**2
    if direction == "reverse":
        values = F.kl_div(
            F.log_softmax(t, dim=-1), F.softmax(s, dim=-1), reduction="none"
        ).sum(dim=-1)
        return values.reshape(-1).mean() * temperature**2
    raise ValueError("direction must be forward or reverse")
