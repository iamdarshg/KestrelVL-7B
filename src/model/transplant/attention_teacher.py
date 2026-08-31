"""Frozen teacher wrapper; no student mutation occurs here."""

from pathlib import Path

import torch


class FrozenTeacher:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_summary(self) -> dict[str, object]:
        return {"training": self.model.training, "parameters": sum(p.numel() for p in self.model.parameters())}

