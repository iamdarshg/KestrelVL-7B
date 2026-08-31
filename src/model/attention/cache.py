"""Small, explicit cache API used by reference attention and generation."""

from dataclasses import dataclass, field

import torch


@dataclass
class LayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    compressed_key: torch.Tensor | None = None

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        compressed_key: torch.Tensor | None = None,
    ) -> None:
        if self.key is None:
            self.key, self.value, self.positions = key, value, positions
            self.compressed_key = compressed_key
            return
        self.key = torch.cat((self.key, key), dim=2)
        self.value = torch.cat((self.value, value), dim=2)
        self.positions = torch.cat((self.positions, positions), dim=1)
        if compressed_key is not None:
            if self.compressed_key is None:
                self.compressed_key = compressed_key
            else:
                self.compressed_key = torch.cat((self.compressed_key, compressed_key), dim=2)


@dataclass
class KestrelCache:
    layers: dict[int, LayerCache] = field(default_factory=dict)

    def get(self, layer_idx: int) -> LayerCache:
        return self.layers.setdefault(layer_idx, LayerCache())

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        compressed_key: torch.Tensor | None = None,
    ) -> None:
        self.get(layer_idx).append(key, value, positions, compressed_key)

    def length(self, layer_idx: int = 0) -> int:
        layer = self.layers.get(layer_idx)
        return 0 if layer is None or layer.key is None else layer.key.shape[2]

    def state_dict(self) -> dict[str, object]:
        return {
            str(idx): {
                "key": item.key,
                "value": item.value,
                "positions": item.positions,
                "compressed_key": item.compressed_key,
            }
            for idx, item in self.layers.items()
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "KestrelCache":
        cache = cls()
        for raw_idx, raw_item in state.items():
            item = raw_item  # type: ignore[assignment]
            cache.layers[int(raw_idx)] = LayerCache(
                item["key"], item["value"], item["positions"], item.get("compressed_key")
            )  # type: ignore[index]
        return cache
