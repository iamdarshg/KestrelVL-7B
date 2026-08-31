import torch


def on_policy_reverse_kl(student_logits: torch.Tensor, specialist_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    student_logp = torch.nn.functional.log_softmax(student_logits.float() / temperature, dim=-1)
    specialist_p = torch.nn.functional.softmax(specialist_logits.float() / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_logp, specialist_p, reduction="batchmean") * temperature**2

