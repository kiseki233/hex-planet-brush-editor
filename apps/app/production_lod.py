from __future__ import annotations

from dataclasses import dataclass

from .brush_lod import BrushLodPolicy
from .zoom_tiers import ZoomTier, ZoomTierPolicy, visible_cells_estimate
from .i18n import t


@dataclass(frozen=True)
class ProductionLodDecision:
    level: int
    mode: str
    candidate_cells: int
    changed: bool
    description: str
    tier: ZoomTier | None = None

    @property
    def detailed(self) -> bool:
        return self.level < 4


class ProductionLodController:
    """Production LOD0-LOD4 policy driven by the six-tier zoom ladder.

    The level used to come from ``BrushLodPolicy`` hysteresis over an estimated
    cell count that was computed differently from the one the editor shows. The
    two estimates disagreed by a constant factor of about 1.74, so the tier the
    interface reported and the level the GPU actually rendered drifted apart by
    one to two steps: the status line claimed 128x128 per-cell texturing from
    zoom 13 while the viewport was still drawing only the far-view surface until
    roughly zoom 21.

    Selecting straight from ``ZOOM_TIERS`` makes the label and the render agree
    by construction, and keeps one definition of what each zoom range means.
    ``BrushLodPolicy`` remains the owner of the texture sizes themselves.
    """

    def __init__(self, initial_level: int = 4) -> None:
        if initial_level < 0 or initial_level > 4:
            raise ValueError("initial_level must be between 0 and 4")
        self.policy = BrushLodPolicy(initial_level)
        # Seed the tier policy so the first update reports ``changed`` honestly.
        self.tier_policy = ZoomTierPolicy(initial_zoom=_zoom_for_level(initial_level))

    @property
    def level(self) -> int:
        return self.policy.level

    def force_level(self, level: int) -> None:
        """Override the tier's choice, e.g. when a frame exceeds a hard budget.

        The tier policy keeps its own state, so the next ordinary update still
        follows the zoom rather than sticking at the forced level.
        """
        if level < 0 or level > 4:
            raise ValueError("level must be between 0 and 4")
        self.policy.level = int(level)

    def update(self, zoom: float, width: int = 1000, height: int = 1000) -> ProductionLodDecision:
        previous = self.policy.level
        tier = self.tier_policy.update(zoom)
        level = 4 if tier.texture_lod is None else int(tier.texture_lod)
        self.policy.level = level
        return ProductionLodDecision(
            level=level,
            mode="detail" if level < 4 else "aggregate",
            candidate_cells=visible_cells_estimate(zoom, width, height),
            changed=level != previous,
            description=(
                f"{tier.name} {BrushLodPolicy.description(level)}"
                if level < 4
                else t("{name} 远景地表", name=tier.name)
            ),
            tier=tier,
        )


def _zoom_for_level(level: int) -> float:
    """Lowest zoom whose tier renders at this texture level."""
    from .zoom_tiers import ZOOM_TIERS

    for tier in reversed(ZOOM_TIERS):
        if (4 if tier.texture_lod is None else tier.texture_lod) == level:
            return tier.zoom_floor
    return 1.0
