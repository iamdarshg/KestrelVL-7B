"""Dynamic 448px tiling and optional InternViT loading hook."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


def dynamic_tiles(image: Any, tile_size: int = 448, max_tiles: int = 16) -> torch.Tensor:
    """Convert a PIL image or CHW/BCHW tensor into padded square tiles.

    Tiling is deterministic and retains the final partial tiles via padding.
    Production preprocessing can replace this function while keeping its
    tensor contract `[tiles, 3, tile_size, tile_size]`.
    """
    if not torch.is_tensor(image):
        import numpy as np

        array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have RGB CHW or BCHW layout")
    _, height, width = image.shape
    rows = max(1, (height + tile_size - 1) // tile_size)
    cols = max(1, (width + tile_size - 1) // tile_size)
    while rows * cols > max_tiles:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    padded = torch.zeros(3, rows * tile_size, cols * tile_size, dtype=image.dtype)
    padded[:, :height, :width] = image
    tiles = padded.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    return tiles.permute(1, 2, 0, 3, 4).reshape(-1, 3, tile_size, tile_size)


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int = 1024, patch_size: int = 14) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, hidden_size, patch_size, patch_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return self.norm(x)


class InternViTEncoder(nn.Module):
    """Loads the exact OpenGVLab model when requested; fallback is test-only."""

    def __init__(self, model_path: str | Path | None = None, hidden_size: int = 1024, freeze: bool = True) -> None:
        super().__init__()
        self.model = None
        self.hidden_size = hidden_size
        if model_path:
            try:
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
                self.hidden_size = int(getattr(self.model.config, "hidden_size", hidden_size))
            except Exception as exc:  # pragma: no cover - depends on optional remote code/weights
                raise RuntimeError(f"could not load InternViT at {model_path}: {exc}") from exc
        else:
            self.model = TinyVisionEncoder(hidden_size)
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values)
        if torch.is_tensor(output):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "hidden_states") and output.hidden_states:
            return output.hidden_states[-1]
        raise TypeError("vision encoder output has no token sequence")

