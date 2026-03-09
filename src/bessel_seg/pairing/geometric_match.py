"""Geometric candidate-pair generation for Bessel beam spot pairing.

Bessel beam physics produces two horizontally-aligned spots per neuron.
Real-data measurements from 689 confirmed pairs (all 10 datasets):
  - Horizontal spacing: mean=42.9 px, median=43.0, std=8.9, range [25, 60]
  - Vertical offset: max observed ≈ 3.2 px (config allows up to 5 px)
  - Intensity ratio: min observed ≈ 0.45 (config allows down to 0.25)
  - Scale similarity: most pairs within 40 % sigma difference

CRITICAL (CLAUDE.md): geometric pairing alone is NOT selective.  Pair counts
increase monotonically with distance range — there is no clear peak at [30-50]
px because random coincidental pairings dominate.  Temporal correlation
MUST be used downstream to validate geometrically-found pairs.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from bessel_seg.config import PairingConfig
from bessel_seg.data_types import Spot

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Module-level constants (all calibrated from real data — CLAUDE.md)
# -------------------------------------------------------------------------
# Maximum search radius: must reach spots at (dx_max, dy_max) from a left spot
# = sqrt(dx_max² + dy_max²).  With dx_max=60, dy_max=5: ≈ 60.2 px.
_SEARCH_RADIUS_MARGIN: float = 1.0   # extra px to catch boundary cases


def _pair_cost(
    left: Spot,
    right: Spot,
    horizontal_dist: float,
    config: PairingConfig,
) -> float:
    """Compute a scalar pairing cost (lower = better match).

    Three equally-important sub-costs, each normalised to roughly [0, 1]:

    1. **distance**: deviation of horizontal_dist from ideal_distance (43.0 px),
       normalised by the half-width of the allowed range.
    2. **intensity**: how far the intensity ratio deviates from 1.0.
    3. **scale**: normalised sigma difference between the two spots.

    Args:
        left: Spot with smaller x coordinate.
        right: Spot with larger x coordinate.
        horizontal_dist: right.x − left.x (already computed by caller).
        config: PairingConfig carrying weights and ideal_distance.

    Returns:
        Weighted sum in [0, ∞).  Pairs with cost > 1.0 are typically spurious.
    """
    w = config.cost_weights  # {"distance": ..., "intensity": ..., "scale": ...}

    # 1. Distance cost: normalised deviation from ideal
    half_range = (config.horizontal_dist_max - config.horizontal_dist_min) / 2.0
    dist_cost = abs(horizontal_dist - config.ideal_distance) / max(half_range, 1.0)

    # 2. Intensity ratio cost: ratio in [0.25, 4.0]; ideal = 1.0
    i_max = max(left.intensity, right.intensity)
    i_min = min(left.intensity, right.intensity)
    if i_max < 1e-9:
        ratio = 1.0
    else:
        ratio = i_min / i_max            # in (0, 1]; ideal = 1.0
    intensity_cost = 1.0 - ratio        # 0 when equal, 1 when ratio→0

    # 3. Scale cost: normalised sigma difference
    sigma_max = max(left.sigma, right.sigma)
    if sigma_max < 1e-9:
        scale_cost = 0.0
    else:
        scale_cost = abs(left.sigma - right.sigma) / sigma_max

    cost = (
        w.get("distance",  0.4) * dist_cost
        + w.get("intensity", 0.3) * intensity_cost
        + w.get("scale",     0.3) * scale_cost
    )
    return float(cost)


def find_candidate_pairs(
    spots: list[Spot],
    config: PairingConfig,
) -> list[tuple[int, int, float]]:
    """Find all geometrically valid candidate spot pairs.

    Uses a :class:`scipy.spatial.KDTree` for efficient neighbour lookup.
    For each ordered pair (i, j) with spots[i].x < spots[j].x, the
    following constraints are applied:

    - ``horizontal_dist ∈ [dist_min, dist_max]``  (default [25, 60] px)
    - ``vertical_offset ≤ vertical_offset_max``   (default 5 px)
    - ``intensity_ratio ∈ [0.25, 4.0]``
    - ``scale_diff / max_sigma ≤ scale_diff_max`` (default 0.4)

    Args:
        spots: List of :class:`~bessel_seg.data_types.Spot` candidates.
        config: :class:`~bessel_seg.config.PairingConfig`.

    Returns:
        List of ``(i, j, cost)`` tuples where ``i`` and ``j`` are indices
        into *spots*, ``spots[i].x < spots[j].x``, and *cost* is the
        weighted pairing cost from :func:`_pair_cost`.
    """
    if len(spots) < 2:
        return []

    # Build KDTree on (y, x) coordinates for efficient radius search
    coords = np.array([[s.y, s.x] for s in spots], dtype=np.float64)
    tree = KDTree(coords)

    search_radius = float(config.horizontal_dist_max) + _SEARCH_RADIUS_MARGIN

    candidates: list[tuple[int, int, float]] = []
    seen: set[frozenset[int]] = set()  # avoid duplicate (i,j) / (j,i) pairs

    for i, spot_i in enumerate(spots):
        # Query all spots within the maximum geometric reach
        neighbour_indices = tree.query_ball_point(
            [spot_i.y, spot_i.x], r=search_radius
        )

        for j in neighbour_indices:
            if j == i:
                continue
            pair_key = frozenset((i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            spot_j = spots[j]

            # Enforce left-to-right ordering (left.x < right.x)
            if spot_i.x < spot_j.x:
                left, right, li, ri = spot_i, spot_j, i, j
            else:
                left, right, li, ri = spot_j, spot_i, j, i

            dx = right.x - left.x     # > 0 by construction
            dy = abs(right.y - left.y)

            # --- Geometric constraints ---
            if not (config.horizontal_dist_min <= dx <= config.horizontal_dist_max):
                continue
            if dy > config.vertical_offset_max:
                continue

            # --- Intensity ratio constraint ---
            i_max = max(left.intensity, right.intensity)
            i_min = min(left.intensity, right.intensity)
            if i_max < 1e-9:
                ratio = 1.0
            else:
                ratio = i_min / i_max
            lo, hi = config.intensity_ratio_range
            # ratio is in (0,1]; convert to symmetric range: 0.25 ≤ ratio ≤ 1/0.25=4
            # Since ratio = min/max ∈ (0,1], check ratio ≥ min(lo, 1/hi)
            min_ratio = min(lo, 1.0 / hi) if hi > 0 else lo
            if ratio < min_ratio:
                continue

            # --- Scale difference constraint ---
            sigma_max = max(left.sigma, right.sigma)
            if sigma_max > 1e-9:
                scale_diff = abs(left.sigma - right.sigma) / sigma_max
                if scale_diff > config.scale_diff_max:
                    continue

            cost = _pair_cost(left, right, dx, config)
            candidates.append((li, ri, cost))

    logger.debug(
        "find_candidate_pairs: %d spots → %d candidate pairs",
        len(spots), len(candidates),
    )
    return candidates
