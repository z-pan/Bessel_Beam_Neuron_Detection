"""Globally-optimal 1-to-1 spot pairing via the Hungarian algorithm.

Given the list of candidate (i, j, cost) triples from
:func:`~bessel_seg.pairing.geometric_match.find_candidate_pairs`, this
module builds a cost matrix and finds the assignment that minimises the
total cost subject to the constraint that every spot appears in at most
one pair (1-to-1 matching).

Implementation uses :func:`scipy.optimize.linear_sum_assignment` which
runs in O(n³) — acceptable for the expected number of candidate spots
(typically < 500 per frame after adaptive filtering).
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from bessel_seg.data_types import Spot, SpotPair

logger = logging.getLogger(__name__)

# Pairs with cost above this threshold are never accepted even if they are
# the "best" assignment available.  Calibrated so that the sum of the three
# worst-case normalised sub-costs (each ≤ 1.0) stays within reach of real
# pairs.  Value 1.5 corresponds to e.g. 50 % distance error + 100 %
# intensity mismatch shared across the three sub-costs.
_MAX_ACCEPTABLE_COST: float = 1.5

# Fill value used for missing entries in the cost matrix (forces the solver
# to prefer any real candidate over an artificial entry)
_LARGE_COST: float = 1e6


def _build_cost_matrix(
    n_spots: int,
    candidate_pairs: list[tuple[int, int, float]],
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float]]:
    """Build a square cost matrix for the Hungarian solver.

    Only spot indices that appear in at least one candidate pair are
    included in the matrix.  This keeps the matrix small when only a
    fraction of all detected spots are geometrically eligible.

    Args:
        n_spots: Total number of spots (determines valid index range).
        candidate_pairs: Output of ``find_candidate_pairs``.

    Returns:
        cost_matrix: (K, K) float64 array where K = number of unique spot
            indices appearing in *candidate_pairs*.  Non-candidate entries
            are filled with ``_LARGE_COST``.
        cost_lookup: Dict mapping ``(i, j)`` → cost for quick lookup.
    """
    if not candidate_pairs:
        return np.empty((0, 0), dtype=np.float64), {}

    # Collect unique spot indices (both left and right of each pair)
    unique_ids = sorted({idx for i, j, _ in candidate_pairs for idx in (i, j)})
    id_to_row = {sid: row for row, sid in enumerate(unique_ids)}
    K = len(unique_ids)

    cost_matrix = np.full((K, K), _LARGE_COST, dtype=np.float64)
    cost_lookup: dict[tuple[int, int], float] = {}

    for i, j, cost in candidate_pairs:
        ri, rj = id_to_row[i], id_to_row[j]
        # Fill symmetrically so the solver can assign either direction
        cost_matrix[ri, rj] = cost
        cost_matrix[rj, ri] = cost
        cost_lookup[(i, j)] = cost
        cost_lookup[(j, i)] = cost

    return cost_matrix, cost_lookup


def solve_optimal_pairing(
    spots: list[Spot],
    candidate_pairs: list[tuple[int, int, float]],
) -> tuple[list[SpotPair], list[Spot]]:
    """Find the globally-optimal 1-to-1 spot pairing (Hungarian algorithm).

    Args:
        spots: Full list of :class:`~bessel_seg.data_types.Spot` objects
            (indices match those in *candidate_pairs*).
        candidate_pairs: List of ``(i, j, cost)`` from
            :func:`~bessel_seg.pairing.geometric_match.find_candidate_pairs`.

    Returns:
        paired: List of :class:`~bessel_seg.data_types.SpotPair` objects.
            In each pair, ``left.x < right.x``.
        orphans: Spots that were not included in any accepted pair.
    """
    if not candidate_pairs or len(spots) < 2:
        logger.debug("solve_optimal_pairing: no candidates — all spots are orphans.")
        return [], list(spots)

    cost_matrix, cost_lookup = _build_cost_matrix(len(spots), candidate_pairs)

    if cost_matrix.size == 0:
        return [], list(spots)

    # Unique spot IDs that appear in the matrix (row-ordered)
    unique_ids = sorted({idx for i, j, _ in candidate_pairs for idx in (i, j)})
    id_to_row = {sid: row for row, sid in enumerate(unique_ids)}
    row_to_id = {row: sid for sid, row in id_to_row.items()}

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    paired_spot_ids: set[int] = set()   # committed spot indices
    committed_pairs: set[tuple[int, int]] = set()  # canonical (lo, hi) keys
    paired: list[SpotPair] = []

    # Build a fast lookup set from candidate_pairs for O(1) membership test
    candidate_key_set: set[tuple[int, int]] = {
        (min(i, j), max(i, j)) for i, j, _ in candidate_pairs
    }

    for ri, ci in zip(row_ind, col_ind):
        if ri == ci:
            # Self-assignment: the solver mapped a row to itself because no
            # real candidate was cheap enough.
            continue

        spot_i_id = row_to_id[ri]
        spot_j_id = row_to_id[ci]

        if spot_i_id == spot_j_id:
            continue

        # Canonical pair key — avoids processing (A,B) and (B,A) separately
        canon = (min(spot_i_id, spot_j_id), max(spot_i_id, spot_j_id))

        # Skip if already committed (symmetric output from linear_sum_assignment)
        if canon in committed_pairs:
            continue

        # Reject if the solver selected a non-candidate pair (_LARGE_COST fill)
        if canon not in candidate_key_set:
            continue

        # Reject if cost is above the acceptance threshold
        actual_cost = cost_lookup.get((spot_i_id, spot_j_id), _LARGE_COST)
        if actual_cost >= _MAX_ACCEPTABLE_COST:
            continue

        # Reject if either spot is already in a previously accepted pair
        if spot_i_id in paired_spot_ids or spot_j_id in paired_spot_ids:
            continue

        spot_a = spots[spot_i_id]
        spot_b = spots[spot_j_id]

        # Enforce left.x < right.x
        if spot_a.x <= spot_b.x:
            left, right = spot_a, spot_b
        else:
            left, right = spot_b, spot_a

        dx = right.x - left.x
        dy = abs(right.y - left.y)
        pair_dist = math.hypot(dx, dy)

        i_max = max(left.intensity, right.intensity)
        i_min = min(left.intensity, right.intensity)
        intensity_ratio = (i_min / i_max) if i_max > 1e-9 else 1.0

        sigma_max = max(left.sigma, right.sigma)
        scale_sim = (
            1.0 - abs(left.sigma - right.sigma) / sigma_max
            if sigma_max > 1e-9
            else 1.0
        )

        pair = SpotPair(
            left=left,
            right=right,
            pair_distance=pair_dist,
            horizontal_distance=dx,
            vertical_offset=dy,
            intensity_ratio=intensity_ratio,
            scale_similarity=scale_sim,
            pair_cost=actual_cost,
        )
        paired.append(pair)
        committed_pairs.add(canon)
        paired_spot_ids.add(spot_i_id)
        paired_spot_ids.add(spot_j_id)

    # All spots not committed to a pair become orphans
    orphans = [s for idx, s in enumerate(spots) if idx not in paired_spot_ids]

    logger.info(
        "solve_optimal_pairing: %d spots → %d pairs, %d orphans",
        len(spots), len(paired), len(orphans),
    )
    return paired, orphans
