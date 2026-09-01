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
import json
import struct

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


_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _read_safetensors_header(state_path: Path) -> tuple[dict[str, Any], int]:
    """Read only the fixed-size safetensors header and return its data origin."""
    with state_path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"safetensors file is truncated: {state_path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header_raw = handle.read(header_size)
    if len(header_raw) != header_size:
        raise ValueError(f"safetensors header is truncated: {state_path}")
    header = json.loads(header_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header, 8 + header_size


def _copy_safetensors_streaming(model: nn.Module, state_path: Path) -> None:
    """Copy a safetensors checkpoint into ``model`` one tensor at a time.

    ``safetensors.torch.load_file`` is normally the right API, but its mmap
    behavior can reserve a very large Windows paging-file budget.  During the
    real 7B Nemotron graft the language model already occupies most of that
    budget, so retain only the small JSON header and one tensor byte range at a
    time.  This is intentionally limited to a single verified local file; the
    normal Hugging Face loader remains the path for other checkpoints.
    """
    header, data_origin = _read_safetensors_header(state_path)
    model_state = model.state_dict()
    expected_keys = set(model_state)
    seen_keys: set[str] = set()
    file_size = state_path.stat().st_size

    with state_path.open("rb") as handle:
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if key not in model_state:
                raise RuntimeError(f"unexpected tensor in vision checkpoint: {key}")
            if not isinstance(metadata, dict):
                raise ValueError(f"invalid metadata for safetensors tensor: {key}")
            dtype_name = metadata.get("dtype")
            source_dtype = _SAFETENSORS_DTYPES.get(dtype_name)
            if source_dtype is None:
                raise ValueError(f"unsupported safetensors dtype {dtype_name!r} for {key}")
            shape = tuple(int(dimension) for dimension in metadata.get("shape", []))
            offsets = metadata.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data offsets for safetensors tensor: {key}")
            start, end = (int(offsets[0]), int(offsets[1]))
            absolute_start = data_origin + start
            absolute_end = data_origin + end
            if start < 0 or end < start or absolute_end > file_size:
                raise ValueError(f"out-of-range data offsets for safetensors tensor: {key}")
            expected_bytes = torch.empty(shape, dtype=source_dtype).numel() * torch.empty(
                (), dtype=source_dtype
            ).element_size()
            if end - start != expected_bytes:
                raise ValueError(
                    f"byte count mismatch for {key}: header={end - start}, expected={expected_bytes}"
                )
            handle.seek(absolute_start)
            raw = handle.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"truncated data for safetensors tensor: {key}")
            source = torch.frombuffer(bytearray(raw), dtype=source_dtype).reshape(shape)
            target = model_state[key]
            if tuple(target.shape) != shape:
                raise RuntimeError(
                    f"shape mismatch for {key}: checkpoint={shape}, model={tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(source.to(dtype=target.dtype, device=target.device))
            seen_keys.add(key)
            del source, raw

    missing_keys = sorted(expected_keys - seen_keys)
    if missing_keys:
        raise RuntimeError(f"vision checkpoint is missing tensors: {missing_keys[:3]}")


def _load_local_model_with_safetensors_fallback(
    model_path: Path,
    torch_dtype: torch.dtype | None,
) -> nn.Module:
    """Load a local custom-code model when Transformers rejects bare metadata.

    Some Hugging Face custom checkpoints, including the published TIPSv2
    artifact, contain a valid safetensors file without the optional
    ``format`` metadata entry.  A few Transformers releases assume that entry
    exists before they dispatch to ``trust_remote_code``.  Instantiating the
    pinned config and loading the verified state dict directly preserves the
    exact weights while avoiding that loader-only incompatibility.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    kwargs: dict[str, object] = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModel.from_config(config, **kwargs)
    state_path = model_path / "model.safetensors"
    _copy_safetensors_streaming(model, state_path)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model


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
            local_path = Path(model_path)
            local_config = local_path / "config.json"
            is_tipsv2 = False
            if local_path.is_dir() and local_config.exists():
                is_tipsv2 = json.loads(local_config.read_text(encoding="utf-8")).get(
                    "model_type"
                ) == "tipsv2"
            try:
                from transformers import AutoModel

                kwargs: dict[str, object] = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                if local_path.is_dir():
                    kwargs["local_files_only"] = True
                if is_tipsv2 and (local_path / "model.safetensors").exists():
                    # Avoid the installed Transformers loader's safetensors
                    # metadata assumption and its eager mmap path.  This is
                    # also materially safer when Nemotron already occupies
                    # most of a small machine's virtual-memory budget.
                    self.model = _load_local_model_with_safetensors_fallback(
                        local_path, torch_dtype
                    )
                else:
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
        model_parameter = next(self.model.parameters(), None)
        if model_parameter is not None and pixel_values.dtype != model_parameter.dtype:
            if pixel_values.is_floating_point() and model_parameter.is_floating_point():
                pixel_values = pixel_values.to(dtype=model_parameter.dtype)
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
