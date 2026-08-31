import torch


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -torch.nn.functional.logsigmoid(beta * margin).mean()

