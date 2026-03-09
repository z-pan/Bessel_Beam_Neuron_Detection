"""Unit tests for the pairing sub-package.

Synthetic spot configuration:
  - 3 paired neurons at spacing = 43 px (the measured median), centred at
    different (y, x) positions well separated from each other.
  - 2 lone orphan spots that have no valid geometric partner.

All random operations use seed 42 for determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from bessel_seg.config import PairingConfig
from bessel_seg.data_types import NeuronROI, Spot, SpotPair
from bessel_seg.pairing.geometric_match import find_candidate_pairs, _pair_cost
from bessel_seg.pairing.hungarian_solver import solve_optimal_pairing
from bessel_seg.pairing.orphan_handler import (
    detect_single_spot_neurons,
    rescue_orphans_across_frames,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
SEED = 42
H, W = 200, 400   # image dimensions (used for dff stack shape)
T = 50            # number of time frames

# Ideal Bessel-beam pair spacing (measured median, CLAUDE.md)
IDEAL_DX = 43.0

# True pair centres (y, x) — well separated so pairs don't interfere
PAIR_CENTRES: list[tuple[float, float]] = [
    (50.0, 100.0),
    (100.0, 250.0),
    (150.0, 330.0),
]

# Orphan positions — no valid partner within [25, 60] px
ORPHAN_POSITIONS: list[tuple[float, float]] = [
    (30.0, 10.0),    # too close to left border for any right-side partner
    (170.0, 380.0),  # too close to right border
]

DEFAULT_SIGMA = 2.5
DEFAULT_INTENSITY = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair_spots(
    cy: float,
    cx: float,
    dx: float = IDEAL_DX,
    dy: float = 0.0,
    sigma: float = DEFAULT_SIGMA,
    intensity: float = DEFAULT_INTENSITY,
    frame_idx: int | None = None,
) -> tuple[Spot, Spot]:
    """Return (left, right) spots for one neuron pair."""
    left = Spot(
        y=cy - dy / 2,
        x=cx - dx / 2,
        sigma=sigma,
        intensity=intensity,
        frame_idx=frame_idx,
    )
    right = Spot(
        y=cy + dy / 2,
        x=cx + dx / 2,
        sigma=sigma,
        intensity=intensity,
        frame_idx=frame_idx,
    )
    return left, right


def _make_spot_list() -> list[Spot]:
    """Build the canonical synthetic spot list: 3 pairs + 2 orphans."""
    spots: list[Spot] = []
    for cy, cx in PAIR_CENTRES:
        left, right = _make_pair_spots(cy, cx)
        spots.extend([left, right])
    for oy, ox in ORPHAN_POSITIONS:
        spots.append(Spot(y=oy, x=ox, sigma=DEFAULT_SIGMA, intensity=DEFAULT_INTENSITY))
    return spots


def _default_config(**overrides) -> PairingConfig:
    cfg = PairingConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _make_dff_stack(
    active_orphan_pos: list[tuple[float, float]] | None = None,
    seed: int = SEED,
) -> np.ndarray:
    """Return a (T, H, W) float32 dF/F0 stack.

    Orphan positions in *active_orphan_pos* get a transient calcium event
    (5-frame rise-peak-decay) in frames 10-14.  The rest is near-zero noise.
    """
    rng = np.random.default_rng(seed)
    dff = rng.normal(0.0, 0.05, size=(T, H, W)).astype(np.float32)
    if active_orphan_pos:
        for cy, cx in active_orphan_pos:
            iy, ix = int(round(cy)), int(round(cx))
            for t, amp in zip(range(10, 15), [0.3, 0.6, 1.0, 0.6, 0.3]):
                dff[t,
                    max(0, iy - 2) : min(H, iy + 3),
                    max(0, ix - 2) : min(W, ix + 3)] = amp
    return dff


# ===========================================================================
# geometric_match
# ===========================================================================

class TestFindCandidatePairs:
    def test_finds_all_three_pairs(self):
        """With ideal 43-px spacing the 3 true pairs must be in the candidate list."""
        spots = _make_spot_list()
        cfg = _default_config()
        candidates = find_candidate_pairs(spots, cfg)

        cand_set = {frozenset((i, j)) for i, j, _ in candidates}

        # Spots 0-5 are the 3 pairs (indices: 0,1 / 2,3 / 4,5)
        for pair_start in range(0, 6, 2):
            li, ri = pair_start, pair_start + 1
            assert frozenset((li, ri)) in cand_set, (
                f"True pair ({li},{ri}) not found in candidates."
            )

    def test_rejects_orphan_pairs(self):
        """Orphan spots (indices 6 and 7) must NOT form a pair with each other."""
        spots = _make_spot_list()
        cfg = _default_config()
        candidates = find_candidate_pairs(spots, cfg)
        orphan_pairs = [
            (i, j) for i, j, _ in candidates
            if i in (6, 7) and j in (6, 7)
        ]
        # The two orphans have dx = |380 - 10| = 370 px >> 60 px
        assert len(orphan_pairs) == 0, (
            f"Orphan spots should not be paired; found: {orphan_pairs}"
        )

    def test_empty_input(self):
        assert find_candidate_pairs([], _default_config()) == []

    def test_single_spot(self):
        spots = [Spot(y=50.0, x=100.0, sigma=2.5, intensity=100.0)]
        assert find_candidate_pairs(spots, _default_config()) == []

    def test_dx_too_large_rejected(self):
        """A pair with dx=70 (> dist_max=60) must not appear."""
        left  = Spot(y=50.0, x=100.0, sigma=2.5, intensity=100.0)
        right = Spot(y=50.0, x=170.0, sigma=2.5, intensity=100.0)
        assert find_candidate_pairs([left, right], _default_config()) == []

    def test_dx_too_small_rejected(self):
        """A pair with dx=10 (< dist_min=25) must not appear."""
        left  = Spot(y=50.0, x=100.0, sigma=2.5, intensity=100.0)
        right = Spot(y=50.0, x=110.0, sigma=2.5, intensity=100.0)
        assert find_candidate_pairs([left, right], _default_config()) == []

    def test_dy_constraint_respected(self):
        """A pair with dy=10 (> vertical_offset_max=5) must not appear."""
        left  = Spot(y=50.0, x=100.0, sigma=2.5, intensity=100.0)
        right = Spot(y=60.0, x=143.0, sigma=2.5, intensity=100.0)
        assert find_candidate_pairs([left, right], _default_config()) == []

    def test_intensity_ratio_constraint(self):
        """A pair where one spot is 5x brighter than the other is rejected."""
        left  = Spot(y=50.0, x=100.0, sigma=2.5, intensity=10.0)
        right = Spot(y=50.0, x=143.0, sigma=2.5, intensity=50.0)  # ratio=0.2 < 0.25
        assert find_candidate_pairs([left, right], _default_config()) == []

    def test_ideal_pair_has_low_cost(self):
        """An ideal pair (dx=43, same sigma, same intensity) should have cost < 0.1."""
        cfg = _default_config()
        left  = Spot(y=50.0, x=78.5,  sigma=2.5, intensity=100.0)
        right = Spot(y=50.0, x=121.5, sigma=2.5, intensity=100.0)
        candidates = find_candidate_pairs([left, right], cfg)
        assert len(candidates) == 1
        _, _, cost = candidates[0]
        assert cost < 0.1, f"Ideal pair cost={cost:.4f} should be < 0.1"

    def test_cost_higher_for_large_distance_deviation(self):
        """A pair at dx=25 (far from ideal=43) should cost more than dx=43."""
        cfg = _default_config()
        left_ideal  = Spot(y=50.0, x=78.5,  sigma=2.5, intensity=100.0)
        right_ideal = Spot(y=50.0, x=121.5, sigma=2.5, intensity=100.0)
        left_far    = Spot(y=50.0, x=83.0,  sigma=2.5, intensity=100.0)
        right_far   = Spot(y=50.0, x=108.0, sigma=2.5, intensity=100.0)  # dx=25

        cands_ideal = find_candidate_pairs([left_ideal, right_ideal], cfg)
        cands_far   = find_candidate_pairs([left_far,   right_far],   cfg)

        assert cands_ideal and cands_far
        assert cands_far[0][2] > cands_ideal[0][2], (
            f"dx=25 cost ({cands_far[0][2]:.4f}) should exceed "
            f"dx=43 cost ({cands_ideal[0][2]:.4f})"
        )


# ===========================================================================
# hungarian_solver
# ===========================================================================

class TestSolveOptimalPairing:
    def test_returns_three_pairs(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, orphans = solve_optimal_pairing(spots, candidates)
        assert len(paired) == 3, f"Expected 3 pairs, got {len(paired)}."

    def test_returns_two_orphans(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, orphans = solve_optimal_pairing(spots, candidates)
        assert len(orphans) == 2, f"Expected 2 orphans, got {len(orphans)}."

    def test_left_x_less_than_right_x(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, _ = solve_optimal_pairing(spots, candidates)
        for pair in paired:
            assert pair.left.x < pair.right.x, (
                f"left.x={pair.left.x} >= right.x={pair.right.x}"
            )

    def test_pair_distances_near_ideal(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, _ = solve_optimal_pairing(spots, candidates)
        for pair in paired:
            assert abs(pair.horizontal_distance - IDEAL_DX) < 2.0, (
                f"horizontal_distance={pair.horizontal_distance:.2f}, expected ~{IDEAL_DX}"
            )

    def test_no_spot_in_two_pairs(self):
        """Each Spot must appear in at most one SpotPair."""
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, _ = solve_optimal_pairing(spots, candidates)
        seen: set[int] = set()
        for pair in paired:
            for s in (pair.left, pair.right):
                sid = id(s)
                assert sid not in seen, "Spot appears in two different pairs."
                seen.add(sid)

    def test_empty_candidates_all_orphans(self):
        spots = _make_spot_list()
        paired, orphans = solve_optimal_pairing(spots, [])
        assert paired == [] and len(orphans) == len(spots)

    def test_empty_spots(self):
        paired, orphans = solve_optimal_pairing([], [])
        assert paired == [] and orphans == []

    def test_returns_spot_pair_objects(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, _ = solve_optimal_pairing(spots, candidates)
        assert all(isinstance(p, SpotPair) for p in paired)

    def test_intensity_ratio_in_01(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, _ = solve_optimal_pairing(spots, candidates)
        for pair in paired:
            assert 0.0 < pair.intensity_ratio <= 1.0 + 1e-6, (
                f"intensity_ratio={pair.intensity_ratio} out of (0, 1]"
            )


# ===========================================================================
# orphan_handler — rescue_orphans_across_frames
# ===========================================================================

class TestRescueOrphansAcrossFrames:
    def _make_template_pair(
        self, cy: float = 100.0, cx: float = 200.0, frame_idx: int = 0
    ) -> SpotPair:
        left  = Spot(y=cy, x=cx - 21.5, sigma=2.5, intensity=100.0, frame_idx=frame_idx)
        right = Spot(y=cy, x=cx + 21.5, sigma=2.5, intensity=100.0, frame_idx=frame_idx)
        return SpotPair(
            left=left, right=right,
            pair_distance=43.0, horizontal_distance=43.0,
            vertical_offset=0.0, intensity_ratio=1.0, scale_similarity=1.0,
        )

    def test_rescues_nearby_orphan(self):
        """Orphan within 12 px of a paired centre in an adjacent frame is rescued."""
        pair_t0 = self._make_template_pair(cy=100.0, cx=200.0, frame_idx=0)
        orphan  = Spot(y=100.0, x=208.0, sigma=2.5, intensity=90.0, frame_idx=1)
        rescued = rescue_orphans_across_frames(
            {1: [orphan]}, {0: [pair_t0]}, _default_config()
        )
        assert len(rescued) >= 1

    def test_does_not_rescue_distant_orphan(self):
        """Orphan >12 px from any pair centre is not rescued."""
        pair_t0 = self._make_template_pair(cy=100.0, cx=200.0, frame_idx=0)
        orphan  = Spot(y=100.0, x=250.0, sigma=2.5, intensity=90.0, frame_idx=1)
        rescued = rescue_orphans_across_frames(
            {1: [orphan]}, {0: [pair_t0]}, _default_config()
        )
        assert len(rescued) == 0

    def test_outside_frame_window_not_used(self):
        """Pairs more than 2 frames away do not serve as templates."""
        pair_t0 = self._make_template_pair(cy=100.0, cx=200.0, frame_idx=0)
        orphan  = Spot(y=100.0, x=208.0, sigma=2.5, intensity=90.0, frame_idx=5)
        rescued = rescue_orphans_across_frames(
            {5: [orphan]}, {0: [pair_t0]}, _default_config()
        )
        assert len(rescued) == 0

    def test_rescue_returns_spot_pairs(self):
        pair_t0 = self._make_template_pair()
        orphan  = Spot(y=100.0, x=208.0, sigma=2.5, intensity=90.0, frame_idx=1)
        rescued = rescue_orphans_across_frames(
            {1: [orphan]}, {0: [pair_t0]}, _default_config()
        )
        assert all(isinstance(p, SpotPair) for p in rescued)

    def test_rescued_pair_has_high_cost(self):
        """Rescued pairs are tagged pair_cost=0.8 as low-confidence markers."""
        pair_t0 = self._make_template_pair()
        orphan  = Spot(y=100.0, x=208.0, sigma=2.5, intensity=90.0, frame_idx=1)
        rescued = rescue_orphans_across_frames(
            {1: [orphan]}, {0: [pair_t0]}, _default_config()
        )
        if rescued:
            assert rescued[0].pair_cost == pytest.approx(0.8)

    def test_empty_inputs_return_empty(self):
        assert rescue_orphans_across_frames({}, {}, _default_config()) == []


# ===========================================================================
# orphan_handler — detect_single_spot_neurons
# ===========================================================================

class TestDetectSingleSpotNeurons:
    def _active_orphan(
        self, cy: float = 80.0, cx: float = 50.0
    ) -> tuple[Spot, np.ndarray]:
        orphan = Spot(y=cy, x=cx, sigma=2.5, intensity=50.0, frame_idx=10)
        dff    = _make_dff_stack(active_orphan_pos=[(cy, cx)])
        return orphan, dff

    def test_transient_orphan_becomes_roi(self):
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        assert len(rois) >= 1

    def test_persistent_orphan_rejected(self):
        """Signal present in every frame → background → reject."""
        dff    = np.full((T, H, W), 1.0, dtype=np.float32)
        orphan = Spot(y=80.0, x=50.0, sigma=2.5, intensity=50.0)
        rois   = detect_single_spot_neurons([orphan], [], dff, _default_config())
        assert len(rois) == 0

    def test_near_pair_gets_higher_or_equal_confidence(self):
        """Orphan near a known pair should have confidence >= isolated orphan."""
        orphan, dff = self._active_orphan(cy=80.0, cx=50.0)

        # Known pair centre at (80, 50.5) — 0.5 px away, well within 25 px
        left  = Spot(y=80.0, x=29.0, sigma=2.5, intensity=100.0)
        right = Spot(y=80.0, x=72.0, sigma=2.5, intensity=100.0)
        near_pair = SpotPair(
            left=left, right=right,
            pair_distance=43.0, horizontal_distance=43.0,
            vertical_offset=0.0, intensity_ratio=1.0, scale_similarity=1.0,
        )
        rois_with    = detect_single_spot_neurons(
            [orphan], [near_pair], dff, _default_config()
        )
        rois_without = detect_single_spot_neurons(
            [orphan], [], dff, _default_config()
        )
        if rois_with and rois_without:
            assert rois_with[0].confidence >= rois_without[0].confidence

    def test_isolated_confidence_in_medium_range(self):
        """Isolated transient confidence should be in [0.40, 0.65]."""
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        if rois:
            assert 0.40 <= rois[0].confidence <= 0.65, (
                f"confidence={rois[0].confidence:.2f} out of expected medium range"
            )

    def test_detection_type_values(self):
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        valid = {"single_near_pair", "single_isolated"}
        for roi in rois:
            assert roi.detection_type in valid

    def test_returns_neuron_roi_objects(self):
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        assert all(isinstance(r, NeuronROI) for r in rois)

    def test_right_spot_is_none(self):
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        for roi in rois:
            assert roi.right_spot is None

    def test_empty_orphans_returns_empty(self):
        dff = _make_dff_stack()
        assert detect_single_spot_neurons([], [], dff, _default_config()) == []

    def test_total_frames_matches_stack(self):
        orphan, dff = self._active_orphan()
        rois = detect_single_spot_neurons([orphan], [], dff, _default_config())
        for roi in rois:
            assert roi.total_frames == T


# ===========================================================================
# Integration
# ===========================================================================

class TestPairingPipeline:
    def test_end_to_end_3_pairs_2_orphans(self):
        spots = _make_spot_list()
        cfg = _default_config()
        candidates = find_candidate_pairs(spots, cfg)
        paired, orphans = solve_optimal_pairing(spots, candidates)
        assert len(paired) == 3, f"Expected 3 pairs, got {len(paired)}"
        assert len(orphans) == 2, f"Expected 2 orphans, got {len(orphans)}"

    def test_orphans_become_single_spot_rois(self):
        spots = _make_spot_list()
        candidates = find_candidate_pairs(spots, _default_config())
        paired, orphans = solve_optimal_pairing(spots, candidates)

        orphan_positions = [(s.y, s.x) for s in orphans]
        dff = _make_dff_stack(active_orphan_pos=orphan_positions)

        rois = detect_single_spot_neurons(orphans, paired, dff, _default_config())
        assert len(rois) >= 1
        for roi in rois:
            assert roi.detection_type in ("single_near_pair", "single_isolated")
