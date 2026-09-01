"""Dynamic 448px tiling and developer-vision encoder adapters.

The public graft contract is a sequence of patch tokens with shape
``[batch, tokens, hidden]``.  InternViT and Google's TIPSv2 custom remote-code
model expose that sequence through different APIs, so this module keeps the
normalization, output extraction, and long-context offload policy in one
place.  TIPSv2 is deliberately an opt-in backend until its local weights have
passed the load and image smoke tests.
"""

from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
from torch import nn


def _extract_token_sequence(output: Any) -> torch.Tensor:
    """Extract patch/spatial tokens from supported vision-model outputs."""
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise TypeError(f"vision tensor must be rank 3, got shape {tuple(output.shape)}")
        return output

    if isinstance(output, dict):
        for key in ("patch_tokens", "x_norm_patchtokens", "last_hidden_state"):
            candidate = output.get(key)
            if torch.is_tensor(candidate):
                return candidate

    # TIPSv2's underlying VisionTransformer returns
    # (class_token, register_tokens, patch_tokens).  Search from the end so
    # the spatial stream wins over the one-token class stream.
    if isinstance(output, (tuple, list)):
        for candidate in reversed(output):
            if torch.is_tensor(candidate) and candidate.ndim == 3:
                return candidate

    # TIPSv2 returns TIPSv2Output(image_features=TIPSv2ImageOutput(...)) and
    # its patch stream is the spatial representation needed by the projector.
    image_features = getattr(output, "image_features", None)
    patch_tokens = getattr(image_features, "patch_tokens", None)
    if torch.is_tensor(patch_tokens):
        return patch_tokens

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(last_hidden_state):
        return last_hidden_state
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        candidate = hidden_states[-1]
        if torch.is_tensor(candidate):
            return candidate
    raise TypeError("vision encoder output has no patch-token sequence")


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

    def __init__(
        self,
        model_path: str | Path | None = None,
        hidden_size: int = 1024,
        freeze: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model = None
        self.hidden_size = hidden_size
        self.backend = "tiny"
        if model_path:
            try:
                from transformers import AutoModel

                kwargs: dict[str, object] = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                if Path(model_path).is_dir():
                    kwargs["local_files_only"] = True
                self.model = AutoModel.from_pretrained(str(model_path), **kwargs)
                config = self.model.config
                self.backend = str(getattr(config, "model_type", "internvit"))
                self.hidden_size = int(
                    getattr(config, "hidden_size", getattr(config, "embed_dim", hidden_size))
                )
            except Exception as exc:  # pragma: no cover - depends on optional remote code/weights
                raise RuntimeError(f"could not load vision encoder at {model_path}: {exc}") from exc
        else:
            self.model = TinyVisionEncoder(hidden_size)
        # The public graft API accepts RGB tensors in [0, 1].  InternViT uses
        # CLIP-style channel normalization; read a model-specific override
        # when remote config exposes one, otherwise use the published
        # OpenCLIP defaults used by the InternViT family.
        model_config = getattr(self.model, "config", None)
        mean = getattr(model_config, "image_mean", None)
        std = getattr(model_config, "image_std", None)
        if self.backend == "tipsv2":
            # TIPSv2's shipped processor explicitly rescales to [0, 1] and
            # does not apply channel normalization.
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        self.register_buffer(
            "pixel_mean",
            torch.tensor(
                mean if mean is not None else [0.48145466, 0.4578275, 0.40821073]
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(
                std if std is not None else [0.26862954, 0.26130258, 0.27577711]
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.last_telemetry: dict[str, object] = {}
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = (pixel_values - self.pixel_mean.to(pixel_values)) / self.pixel_std.to(pixel_values)
        if self.backend == "tipsv2" and hasattr(self.model, "vision_encoder"):
            # TIPSv2's public encode_image method is decorated with
            # ``torch.no_grad``.  Calling the underlying ViT preserves the
            # staged-unfreezing contract for last4/upper12/all training.
            output = self.model.vision_encoder(pixel_values)
        else:
            output = self.model(pixel_values)
        return _extract_token_sequence(output)

    def encode_with_policy(
        self,
        pixel_values: torch.Tensor,
        language_device: torch.device,
        offload_to_cpu: bool = False,
    ) -> torch.Tensor:
        """Encode once, optionally keeping the frozen ViT on CPU.

        The returned sequence is placed on ``language_device`` exactly once;
        callers should project it there and reuse the projected tokens across
        all text chunks.  This avoids a per-chunk GPU/CPU shuttle for long
        multimodal prompts.
        """
        vision_device = torch.device("cpu") if offload_to_cpu else language_device
        if next(self.parameters(), torch.empty(0)).device != vision_device:
            self.to(vision_device)
        requires_grad = any(parameter.requires_grad for parameter in self.parameters())
        context = nullcontext() if requires_grad else torch.no_grad()
        with context:
            encoded = self(pixel_values.to(vision_device, non_blocking=True))
        encoded = encoded.to(language_device, non_blocking=True)
        self.last_telemetry = {
            "vision_device": str(vision_device),
            "language_device": str(language_device),
            "offloaded": offload_to_cpu,
            "encoded_once": True,
            "tiles": int(pixel_values.shape[0]),
            "tokens": int(encoded.shape[-2]),
        }
        return encoded
