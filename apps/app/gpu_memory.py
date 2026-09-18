from __future__ import annotations


DEFAULT_GPU_TEXTURE_BUDGET_BYTES = 3 * 1024 * 1024 * 1024
DEFAULT_TEXTURE_LAYER_HEADROOM = 512
MINIMUM_TEXTURE_LAYERS = 2
MINIMUM_STREAM_TEXTURE_LAYERS = 1024


def texture_layer_bytes(padded_size: int) -> int:
    side = int(padded_size)
    if side <= 0:
        raise ValueError("padded_size must be positive")
    return side * side * 4


def budgeted_texture_layer_limit(
    padded_size: int,
    *,
    budget_bytes: int = DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
    configured_limit: int = 4096,
) -> int:
    """Return the RGBA8 array depth that fits both byte and layer limits."""

    budget = max(texture_layer_bytes(padded_size) * MINIMUM_TEXTURE_LAYERS, int(budget_bytes))
    limit = max(MINIMUM_TEXTURE_LAYERS, int(configured_limit))
    return max(
        MINIMUM_TEXTURE_LAYERS,
        min(limit, budget // texture_layer_bytes(padded_size)),
    )


def catalog_texture_layer_limit(
    active_brush_count: int,
    *,
    padded_size: int,
    budget_bytes: int = DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
    configured_limit: int = 4096,
    headroom: int = DEFAULT_TEXTURE_LAYER_HEADROOM,
) -> int:
    """Cap residency to the current catalog plus growth room and the byte budget."""

    desired = max(
        MINIMUM_STREAM_TEXTURE_LAYERS,
        int(active_brush_count) + MINIMUM_TEXTURE_LAYERS + max(0, int(headroom)),
    )
    return min(
        desired,
        budgeted_texture_layer_limit(
            padded_size,
            budget_bytes=budget_bytes,
            configured_limit=configured_limit,
        ),
    )


def next_texture_capacity(required: int, current: int, maximum: int) -> int:
    """Grow geometrically, clamping the final step to the real memory budget."""

    required = max(MINIMUM_TEXTURE_LAYERS, int(required))
    maximum = max(MINIMUM_TEXTURE_LAYERS, int(maximum))
    if required > maximum:
        raise ValueError(
            f"required texture layers exceed the configured limit: {required} / {maximum}"
        )
    current = max(0, int(current))
    if current <= 0:
        target = 32
    else:
        target = max(current + 1, current * 2)
    return min(maximum, max(required, target))
