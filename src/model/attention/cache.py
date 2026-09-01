"""Bounded, explicit KV cache for the reference attention implementation.

The first prototype kept the complete token K/V history and appended with
``torch.cat`` on every decode step.  That is both quadratic in allocator work
and incompatible with million-token inference.  The cache below keeps only a
bounded local window plus append-only compressed chunks.  Materializing the
compressed history is an explicit compatibility operation; the attention
modules use the chunk lists directly.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class CompressedStateStore:
    """Append-only compressed K/V chunks with no historical concatenation."""

    key_chunks: list[torch.Tensor] = field(default_factory=list)
    value_chunks: list[torch.Tensor] = field(default_factory=list)
    position_chunks: list[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("compressed key/value must both have shape [B, KV, M, D]")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape != (key.shape[0], key.shape[2]):
            raise ValueError("compressed positions must have shape [B, M]")
        if key.shape[2] == 0:
            return
        # Store references, rather than repeatedly reallocating the complete
        # history.  The caller decides whether these tensors should retain an
        # autograd graph (full-recompute mode) or be detached (decode/cache
        # mode).
        self.key_chunks.append(key)
        self.value_chunks.append(value)
        self.position_chunks.append(positions)

    @property
    def token_count(self) -> int:
        return sum(int(chunk.shape[2]) for chunk in self.key_chunks)

    def iter_chunks(self):
        return zip(self.key_chunks, self.value_chunks, self.position_chunks)

    def materialize(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.key_chunks:
            return None, None, None
        return (
            torch.cat(self.key_chunks, dim=2),
            torch.cat(self.value_chunks, dim=2),
            torch.cat(self.position_chunks, dim=1),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "key_chunks": self.key_chunks,
            "value_chunks": self.value_chunks,
            "position_chunks": self.position_chunks,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CompressedStateStore":
        # The old cache format had one materialized compressed_key tensor and
        # no compressed values/positions.  New checkpoints use chunk lists.
        raw_keys = state.get("key_chunks", [])
        raw_values = state.get("value_chunks", [])
        raw_positions = state.get("position_chunks", [])
        return cls(list(raw_keys), list(raw_values), list(raw_positions))  # type: ignore[arg-type]


@dataclass
class LayerCache:
    # ``key/value/positions`` are the bounded local state.  These names remain
    # public for compatibility with existing callers and inspection tools.
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    compressed_key: torch.Tensor | None = None
    compressed_value: torch.Tensor | None = None
    compressed_positions: torch.Tensor | None = None
    local_window: int = 128
    next_position: int = 0
    pending_key: torch.Tensor | None = None
    pending_value: torch.Tensor | None = None
    pending_positions: torch.Tensor | None = None
    compressed: CompressedStateStore = field(default_factory=CompressedStateStore)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        compressed_key: torch.Tensor | None = None,
        compressed_value: torch.Tensor | None = None,
        compressed_positions: torch.Tensor | None = None,
        local_window: int | None = None,
    ) -> None:
        """Append current-token state while retaining only bounded local K/V."""
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("key/value must both have shape [B, KV, T, D]")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape != (key.shape[0], key.shape[2]):
            raise ValueError("positions must have shape [B, T]")
        if local_window is not None:
            if local_window < 1:
                raise ValueError("local_window must be positive")
            self.local_window = local_window

        if self.key is None:
            local_key, local_value, local_positions = key, value, positions
        else:
            # This concatenation is bounded by local_window + current chunk;
            # it never touches the compressed historical state.
            local_key = torch.cat((self.key, key), dim=2)
            local_value = torch.cat((self.value, value), dim=2)  # type: ignore[arg-type]
            local_positions = torch.cat((self.positions, positions), dim=1)  # type: ignore[arg-type]
        keep = min(self.local_window, local_key.shape[2])
        self.key = local_key[:, :, -keep:]
        self.value = local_value[:, :, -keep:]
        self.positions = local_positions[:, -keep:]
        self.next_position = max(self.next_position, int(positions.max().detach().cpu()) + 1)

        if compressed_key is not None:
            if compressed_value is None:
                compressed_value = compressed_key
            if compressed_positions is None:
                # This fallback is only for old callers that supplied a
                # token-level compressed_key.  New attention code always
                # supplies group-end positions explicitly.
                compressed_positions = positions[:, -compressed_key.shape[2] :]
            self.compressed.append(compressed_key, compressed_value, compressed_positions)
            self.compressed_key = None
            self.compressed_value = None
            self.compressed_positions = None

    @property
    def compressed_chunks(self):
        return self.compressed.iter_chunks()

    @property
    def compressed_token_count(self) -> int:
        return self.compressed.token_count

    def materialize_compressed(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.compressed.materialize()

    def legacy_compressed_key(self) -> torch.Tensor | None:
        key, _, _ = self.materialize_compressed()
        return key if key is not None else self.compressed_key

    def state_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "value": self.value,
            "positions": self.positions,
            "local_window": self.local_window,
            "next_position": self.next_position,
            "pending_key": self.pending_key,
            "pending_value": self.pending_value,
            "pending_positions": self.pending_positions,
            "compressed": self.compressed.state_dict(),
            # Keep legacy fields so old inspection code can identify a cache
            # with no compressed chunks without forcing materialization.
            "compressed_key": self.compressed_key,
            "compressed_value": self.compressed_value,
            "compressed_positions": self.compressed_positions,
        }


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
        compressed_value: torch.Tensor | None = None,
        compressed_positions: torch.Tensor | None = None,
        local_window: int | None = None,
    ) -> None:
        self.get(layer_idx).append(
            key,
            value,
            positions,
            compressed_key,
            compressed_value,
            compressed_positions,
            local_window,
        )

    def length(self, layer_idx: int = 0) -> int:
        layer = self.layers.get(layer_idx)
        return 0 if layer is None else layer.next_position

    def state_dict(self) -> dict[str, object]:
        return {str(idx): item.state_dict() for idx, item in self.layers.items()}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "KestrelCache":
        cache = cls()
        for raw_idx, raw_item in state.items():
            item = raw_item  # type: ignore[assignment]
            compressed = item.get("compressed")  # type: ignore[union-attr]
            if compressed is None:
                old_key = item.get("compressed_key")  # type: ignore[union-attr]
                old_value = item.get("compressed_value")  # type: ignore[union-attr]
                old_positions = item.get("compressed_positions")  # type: ignore[union-attr]
                compressed_store = CompressedStateStore()
                if old_key is not None:
                    if old_value is None:
                        old_value = old_key
                    if old_positions is None:
                        old_positions = item["positions"][:, -old_key.shape[2] :]  # type: ignore[index]
                    compressed_store.append(old_key, old_value, old_positions)
            else:
                compressed_store = CompressedStateStore.from_state_dict(compressed)  # type: ignore[arg-type]
            cache.layers[int(raw_idx)] = LayerCache(
                key=item.get("key"),  # type: ignore[union-attr]
                value=item.get("value"),  # type: ignore[union-attr]
                positions=item.get("positions"),  # type: ignore[union-attr]
                compressed_key=item.get("compressed_key"),  # type: ignore[union-attr]
                compressed_value=item.get("compressed_value"),  # type: ignore[union-attr]
                compressed_positions=item.get("compressed_positions"),  # type: ignore[union-attr]
                local_window=int(item.get("local_window", 128)),  # type: ignore[union-attr]
                next_position=int(item.get("next_position", 0)),  # type: ignore[union-attr]
                pending_key=item.get("pending_key"),  # type: ignore[union-attr]
                pending_value=item.get("pending_value"),  # type: ignore[union-attr]
                pending_positions=item.get("pending_positions"),  # type: ignore[union-attr]
                compressed=compressed_store,
            )
            if cache.layers[int(raw_idx)].next_position == 0 and item.get("positions") is not None:  # type: ignore[union-attr]
                cache.layers[int(raw_idx)].next_position = int(item["positions"].max().cpu()) + 1  # type: ignore[index]
        return cache
