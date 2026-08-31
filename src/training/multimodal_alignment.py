import torch


def projector_alignment_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(projected.float(), target.float())

