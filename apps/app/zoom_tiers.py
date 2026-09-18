from __future__ import annotations

import math
from dataclasses import dataclass
from .i18n import t

MIN_ZOOM = 0.8
MAX_ZOOM = 512.0
MIN_BRUSH_DIAMETER = 1
MAX_BRUSH_DIAMETER = 500
BASE_DRAG_RADIANS_PER_PIXEL = 0.008

# The Tk sphere viewport draws the planet at ``0.42 * min(width, height) * zoom``
# pixels of radius.  The visible spherical cap therefore has an angular radius of
# ``asin(0.5 / 0.42 / zoom)`` once the sphere is larger than the window.
VIEWPORT_RADIUS_FACTOR = 0.42
PRODUCTION_CELL_COUNT = 10_080_162


class ZoomTierError(ValueError):
    pass


def drag_radians_per_pixel(zoom: float) -> float:
    """Keep the apparent rotation speed stable while zooming.

    The projected sphere radius grows linearly with ``zoom``. Dividing the
    angular drag step by the same factor means one mouse pixel moves the map by
    roughly the same number of screen pixels at every zoom level.
    """

    value = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
    return BASE_DRAG_RADIANS_PER_PIXEL / max(1.0, value)


@dataclass(frozen=True)
class ZoomTier:
    """One rung of the zoom-out ladder.

    Every tier is editable.  What changes between tiers is how the surface is
    drawn and how coarse the visual feedback is, never the picking precision:
    the screen-to-CellId picker resolves a single cell at every zoom level.
    """

    index: int
    name: str
    zoom_floor: float
    render: str
    texture_lod: int | None
    feedback_cells: int
    min_diameter: int
    suggested_diameter: int
    max_diameter: int
    description: str

    def clamp_diameter(self, diameter: int) -> int:
        value = int(diameter)
        if value < self.min_diameter:
            return self.min_diameter
        if value > self.max_diameter:
            return self.max_diameter
        return value


# Ordered from the closest tier to the whole-planet tier.  Each floor is roughly
# 2.9x apart in zoom, which is ~8.5x apart in visible cells, so the ladder is
# uniform on a logarithmic scale instead of crowding four texture levels into the
# near view and leaving one bucket for everything else.
ZOOM_TIERS: tuple[ZoomTier, ...] = (
    ZoomTier(
        index=0,
        name=t("L0 原图"),
        zoom_floor=110.0,
        render="cell_texture",
        texture_lod=0,
        feedback_cells=1,
        min_diameter=1,
        suggested_diameter=2,
        max_diameter=35,
        description=t("512×512 逐格纹理"),
    ),
    ZoomTier(
        index=1,
        name=t("L1 精细"),
        zoom_floor=38.0,
        render="cell_texture",
        texture_lod=1,
        feedback_cells=1,
        min_diameter=1,
        suggested_diameter=6,
        max_diameter=100,
        description=t("256×256 逐格纹理"),
    ),
    ZoomTier(
        index=2,
        name=t("L2 常用"),
        zoom_floor=13.0,
        render="cell_texture",
        texture_lod=2,
        feedback_cells=1,
        min_diameter=1,
        suggested_diameter=19,
        max_diameter=300,
        description=t("128×128 逐格纹理"),
    ),
    ZoomTier(
        index=3,
        name=t("L3 区域"),
        zoom_floor=4.5,
        render="cell_color",
        texture_lod=3,
        feedback_cells=1,
        min_diameter=1,
        suggested_diameter=54,
        max_diameter=500,
        description=t("64×64 逐格纹理／逐格代表色"),
    ),
    ZoomTier(
        index=4,
        name=t("L4 地块"),
        zoom_floor=1.6,
        render="chunk",
        texture_lod=None,
        feedback_cells=16,
        min_diameter=32,
        suggested_diameter=150,
        max_diameter=500,
        description=t("按逻辑区块聚合显示（258 格/块）"),
    ),
    ZoomTier(
        index=5,
        name=t("L5 全球"),
        zoom_floor=MIN_ZOOM,
        render="surface",
        texture_lod=None,
        feedback_cells=5,
        min_diameter=64,
        suggested_diameter=300,
        max_diameter=500,
        description=t("整球缩略地表／多层聚合远景"),
    ),
)

TIER_BY_INDEX = {tier.index: tier for tier in ZOOM_TIERS}


def tier_ceiling(tier: ZoomTier) -> float:
    """Return the zoom at which this tier gives way to the next closer tier."""
    if tier.index == 0:
        return math.inf
    return TIER_BY_INDEX[tier.index - 1].zoom_floor


def tier_for_zoom(zoom: float) -> ZoomTier:
    value = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
    for tier in ZOOM_TIERS:
        if value >= tier.zoom_floor:
            return tier
    return ZOOM_TIERS[-1]


def visible_cells_estimate(
    zoom: float,
    width: int = 1000,
    height: int = 1000,
    *,
    cell_count: int = PRODUCTION_CELL_COUNT,
) -> int:
    """Estimate how many cells the viewport covers at this zoom.

    The viewport shows a spherical cap.  Once the projected planet is smaller
    than the window the cap saturates at one hemisphere.
    """
    zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
    width = max(1, int(width))
    height = max(1, int(height))
    radius_px = VIEWPORT_RADIUS_FACTOR * min(width, height) * zoom
    half_extent = min(width, height) / 2.0
    ratio = half_extent / radius_px
    if ratio >= 1.0:
        angle = math.pi / 2.0
    else:
        angle = math.asin(ratio)
    return int(cell_count * (1.0 - math.cos(angle)) / 2.0)


def cell_pixels(zoom: float, width: int = 1000, height: int = 1000, *, frequency: int = 1004) -> float:
    """Screen pixels covered by one cell at this zoom."""
    zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
    radius_px = VIEWPORT_RADIUS_FACTOR * min(max(1, int(width)), max(1, int(height))) * zoom
    # Adjacent cell centers are separated by about 1.2 / frequency radians.
    return radius_px * (1.2 / max(1, int(frequency)))


class ZoomTierPolicy:
    """Select the active tier from the zoom with symmetric hysteresis.

    Without hysteresis a zoom sitting exactly on a boundary would flip the render
    mode and the brush clamp on every wheel notch.
    """

    def __init__(self, initial_zoom: float = 1.0, hysteresis: float = 0.06) -> None:
        if hysteresis < 0.0 or hysteresis >= 1.0:
            raise ZoomTierError("hysteresis must be within [0, 1)")
        self.hysteresis = float(hysteresis)
        self.tier = tier_for_zoom(initial_zoom)

    def update(self, zoom: float) -> ZoomTier:
        value = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
        current = self.tier
        low = current.zoom_floor / (1.0 + self.hysteresis)
        high = tier_ceiling(current) * (1.0 + self.hysteresis)
        if low <= value < high:
            return current
        self.tier = tier_for_zoom(value)
        return self.tier

    def clamp_diameter(self, diameter: int) -> int:
        return self.tier.clamp_diameter(diameter)
