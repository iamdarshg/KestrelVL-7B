import torch


def fake_quantize(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    flat = x.reshape(-1, group_size) if x.numel() % group_size == 0 else x.reshape(1, -1)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((flat / scale).round().clamp(-qmax, qmax) * scale).reshape_as(x)

