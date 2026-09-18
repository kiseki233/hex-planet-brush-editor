from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


class TextureResidencyError(ValueError):
    pass


@dataclass(frozen=True)
class TextureSlot:
    layer: int
    key: str
    pinned: bool
    references: int
    last_used_tick: int


@dataclass(frozen=True)
class TextureAllocation:
    layer: int
    key: str
    reused: bool
    replaced_key: str | None


@dataclass(frozen=True)
class TextureResidencyUpdate:
    released_layers: tuple[int, ...]
    released_keys: tuple[str, ...]
    active_layers: int
    reusable_layers: int


class TextureResidencyManager:
    """Logical GPU texture-array residency with ref-counted slot reuse.

    Layers 0 and 1 are normally pinned for the empty and missing textures. Brush
    layers are released after their reference count reaches zero. A small grace
    period avoids upload churn while the camera crosses a chunk boundary.
    """

    def __init__(
        self,
        pinned_keys: Iterable[str] = (),
        *,
        maximum_layers: int = 4096,
        grace_ticks: int = 2,
        retain_unused: bool = False,
    ) -> None:
        if maximum_layers < 2:
            raise TextureResidencyError("maximum_layers must be at least 2")
        if grace_ticks < 0:
            raise TextureResidencyError("grace_ticks must not be negative")
        self.maximum_layers = int(maximum_layers)
        self.grace_ticks = int(grace_ticks)
        self.retain_unused = bool(retain_unused)
        self.tick = 0
        self.key_to_layer: dict[str, int] = {}
        self.layer_to_key: dict[int, str] = {}
        self.pinned: set[str] = set()
        self.references: Counter[int] = Counter()
        self.last_used: dict[int, int] = {}
        self.free_layers: list[int] = []
        for key in pinned_keys:
            self._append_pinned(str(key))

    def _append_pinned(self, key: str) -> None:
        if key in self.key_to_layer:
            return
        layer = len(self.layer_to_key)
        if layer >= self.maximum_layers:
            raise TextureResidencyError("Pinned textures exceed maximum layer count")
        self.key_to_layer[key] = layer
        self.layer_to_key[layer] = key
        self.pinned.add(key)
        self.references[layer] = 1
        self.last_used[layer] = self.tick

    def allocate(self, key: str) -> TextureAllocation:
        key = str(key)
        existing = self.key_to_layer.get(key)
        if existing is not None:
            self.last_used[existing] = self.tick
            return TextureAllocation(existing, key, False, None)
        replaced_key: str | None = None
        reused = bool(self.free_layers)
        if reused:
            layer = self.free_layers.pop(0)
            replaced_key = self.layer_to_key.get(layer)
            if replaced_key is not None:
                self.key_to_layer.pop(replaced_key, None)
        else:
            layer = len(self.layer_to_key)
            if layer >= self.maximum_layers:
                self._evict_one()
                if not self.free_layers:
                    raise TextureResidencyError(
                        f"GPU texture-array layer limit reached ({self.maximum_layers})"
                    )
                layer = self.free_layers.pop(0)
                replaced_key = self.layer_to_key.get(layer)
                if replaced_key is not None:
                    self.key_to_layer.pop(replaced_key, None)
                reused = True
        self.key_to_layer[key] = layer
        self.layer_to_key[layer] = key
        self.references[layer] = 0
        self.last_used[layer] = self.tick
        return TextureAllocation(layer, key, reused, replaced_key)

    def synchronize(self, layer_references: Iterable[int]) -> TextureResidencyUpdate:
        self.tick += 1
        counts = Counter(int(layer) for layer in layer_references)
        for layer, key in self.layer_to_key.items():
            if key in self.pinned:
                self.references[layer] = max(1, counts.get(layer, 0))
                self.last_used[layer] = self.tick
                continue
            count = counts.get(layer, 0)
            self.references[layer] = count
            if count > 0:
                self.last_used[layer] = self.tick

        released_layers: list[int] = []
        released_keys: list[str] = []
        if not self.retain_unused:
            for layer, key in sorted(tuple(self.layer_to_key.items())):
                if key in self.pinned or self.references.get(layer, 0) > 0:
                    continue
                if self.tick - self.last_used.get(layer, 0) <= self.grace_ticks:
                    continue
                self.key_to_layer.pop(key, None)
                released_layers.append(layer)
                released_keys.append(key)
                if layer not in self.free_layers:
                    self.free_layers.append(layer)
        self.free_layers.sort()
        return TextureResidencyUpdate(
            released_layers=tuple(released_layers),
            released_keys=tuple(released_keys),
            active_layers=sum(
                1
                for layer, key in self.layer_to_key.items()
                if key in self.pinned or self.references.get(layer, 0) > 0
            ),
            reusable_layers=len(self.free_layers),
        )

    def touch_key(self, key: str) -> None:
        layer = self.key_to_layer.get(key)
        if layer is not None:
            self.last_used[layer] = self.tick

    def _evict_one(self) -> None:
        candidates = [
            (self.last_used.get(layer, 0), layer, key)
            for layer, key in self.layer_to_key.items()
            if key not in self.pinned and self.references.get(layer, 0) == 0
        ]
        if not candidates:
            return
        _, layer, key = min(candidates)
        self.key_to_layer.pop(key, None)
        if layer not in self.free_layers:
            self.free_layers.append(layer)
            self.free_layers.sort()

    def slot(self, layer: int) -> TextureSlot | None:
        key = self.layer_to_key.get(int(layer))
        if key is None:
            return None
        return TextureSlot(
            layer=int(layer),
            key=key,
            pinned=key in self.pinned,
            references=self.references.get(int(layer), 0),
            last_used_tick=self.last_used.get(int(layer), 0),
        )

    def layer_for_key(self, key: str) -> int | None:
        return self.key_to_layer.get(str(key))

    def active_slots(self) -> tuple[TextureSlot, ...]:
        return tuple(
            slot
            for layer in sorted(self.layer_to_key)
            if (slot := self.slot(layer)) is not None
            and (slot.pinned or slot.references > 0)
        )
