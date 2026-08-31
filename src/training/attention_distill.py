"""Stage A/B losses for mixer and hidden-state recovery."""

import torch

from model.transplant.representation_alignment import distillation_loss, representation_loss


def attention_imitation_loss(student_mixer: torch.Tensor, teacher_mixer: torch.Tensor) -> torch.Tensor:
    return representation_loss(student_mixer, teacher_mixer, "mse") + representation_loss(student_mixer, teacher_mixer, "cosine")


def hidden_state_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    a, b, c, d = weights
    return a * distillation_loss(student_logits, teacher_logits, direction="forward") + b * representation_loss(student_hidden, teacher_hidden, "normalized") + c * representation_loss(student_hidden, teacher_hidden, "cosine") + d * representation_loss(student_hidden, teacher_hidden, "mse")

