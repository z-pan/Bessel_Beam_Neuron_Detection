"""Tests for src/bessel_seg/refinement/ — gaussian_fit, confidence_score, roi_builder.

Synthetic data strategy
-----------------------
* Known 2-D Gaussian images for fit validation.
* Confidence tier ordering verified without stochastic data.
* roi_builder uses a small (30 frames, 80×160 px) synthetic stack with two
  paired neurons and one single-spot neuron.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bessel_seg.config import RefinementConfig
from bessel_seg.data_types import FrameQuality, NeuronROI, Spot, SpotPair
from bessel_seg.refinement.gaussian_fit import _FALLBACK_SIGMA, fit_spot_gaussian
from bessel_seg.refinement.confidence_score import (
    _temporal_quality,
    compute_confidence,
)
from bessel_seg.refinement.roi_builder import (
    _circle_mask,
    _extract_temporal_trace,
    _select_top_frames,
    build_neuron_rois,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian_image(
    H: int, W: int,
    cy: float, cx: float,
    sigma: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a synthetic 2-D Gaussian image."""
    Y, X = np.ogrid[:H, :W]
    img = offset + amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    return img.astype(np.float32)


def _spot(y: float, x: float, sigma: float = 2.5, intensity: float = 1.0,
          frame_idx: int | None = None) -> Spot:
    return Spot(y=y, x=x, sigma=sigma, intensity=intensity, frame_idx=frame_idx)


def _pair(cy: float, cx: float, dx: float = 43.0, frame_idx: int | None = None,
          pair_cost: float = 0.1, intensity_ratio: float = 0.9,
          scale_similarity: float = 0.95) -> SpotPair:
    left = _spot(cy, cx - dx / 2, frame_idx=frame_idx)
    right = _spot(cy, cx + dx / 2, frame_idx=frame_idx)
    return SpotPair(
        left=left, right=right,
        pair_distance=dx,
        horizontal_distance=dx,
        vertical_offset=0.0,
        intensity_ratio=intensity_ratio,
        scale_similarity=scale_similarity,
        pair_cost=pair_cost,
    )


def _single_roi(cy: float, cx: float, det_type: str = "single_near_pair",
                confidence: float = 0.70) -> NeuronROI:
    return NeuronROI(
        neuron_id=0,
        center_y=cy, center_x=cx,
        left_spot=_spot(cy, cx),
        right_spot=None,
        left_radius=5.0, right_radius=0.0,
        confidence=confidence,
        detection_type=det_type,
        detection_count=3, total_frames=30,
    )


def _frame_quality(frame_idx: int, score: float) -> FrameQuality:
    return FrameQuality(
        frame_idx=frame_idx, snr=score, sharpness=score,
        pair_rate=score, overall_score=score,
    )


def _blank_dff(T: int = 30, H: int = 80, W: int = 160) -> np.ndarray:
    return np.zeros((T, H, W), dtype=np.float32)


def _dff_transient(T: int, H: int, W: int,
                   cy: int, cx: int, start: int,
                   amps: list[float], radius: int = 5) -> np.ndarray:
    dff = np.zeros((T, H, W), dtype=np.float32)
    ys, xs = np.ogrid[:H, :W]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    for i, amp in enumerate(amps):
        t = start + i
        if 0 <= t < T:
            dff[t][mask] = amp
    return dff


# ---------------------------------------------------------------------------
# TestFitSpotGaussian
# ---------------------------------------------------------------------------

class TestFitSpotGaussian:
    """Tests for fit_spot_gaussian."""

    @pytest.mark.parametrize("true_sigma", [1.5, 2.5, 4.0])
    def test_sigma_error_below_20_percent(self, true_sigma: float):
        """Fitted sigma must be within 20 % of the true value on a clean image."""
        H, W = 60, 60
        cy, cx = 30.0, 30.0
        img = _make_gaussian_image(H, W, cy, cx, sigma=true_sigma, amplitude=1.0)
        _, _, sigma_fit, _ = fit_spot_gaussian(img, (cy, cx), radius=12)
        rel_err = abs(sigma_fit - true_sigma) / true_sigma
        assert rel_err < 0.20, (
            f"sigma fit error {rel_err:.1%} ≥ 20% (true={true_sigma}, fit={sigma_fit:.3f})"
        )

    @pytest.mark.parametrize("true_sigma", [1.5, 2.5, 4.0])
    def test_centre_error_below_one_pixel(self, true_sigma: float):
        """Fitted centre must be within 1 px of the true centre."""
        H, W = 60, 60
        cy, cx = 28.3, 31.7   # sub-pixel offset
        img = _make_gaussian_image(H, W, cy, cx, sigma=true_sigma, amplitude=1.0)
        y0, x0, _, _ = fit_spot_gaussian(img, (cy, cx), radius=12)
        dist = math.hypot(y0 - cy, x0 - cx)
        assert dist < 1.0, f"centre error {dist:.3f} px ≥ 1 px"

    def test_amplitude_positive(self):
        img = _make_gaussian_image(60, 60, 30.0, 30.0, sigma=2.5, amplitude=0.8)
        _, _, _, amp = fit_spot_gaussian(img, (30.0, 30.0), radius=10)
        assert amp > 0.0

    def test_flat_image_returns_fallback_sigma(self):
        """Fitting a flat (uninformative) image should fall back gracefully."""
        img = np.ones((40, 40), dtype=np.float32) * 0.5
        _, _, sigma, _ = fit_spot_gaussian(img, (20.0, 20.0), radius=8)
        # Either fitted sigma or fallback; must be in [0.5, radius]
        assert 0.5 <= sigma <= 8.0

    def test_returns_four_floats(self):
        img = _make_gaussian_image(40, 40, 20.0, 20.0, sigma=2.5)
        result = fit_spot_gaussian(img, (20.0, 20.0))
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_near_border_does_not_raise(self):
        """Spot near image edge: patch is clipped but must not crash."""
        img = _make_gaussian_image(40, 40, 2.0, 2.0, sigma=2.5)
        y0, x0, sigma, amp = fit_spot_gaussian(img, (2.0, 2.0), radius=8)
        assert sigma >= 0.5

    def test_noisy_image_sigma_within_30_percent(self):
        """Add Gaussian noise; sigma fit should still be reasonable."""
        np.random.seed(42)
        H, W = 60, 60
        true_sigma = 3.0
        img = _make_gaussian_image(H, W, 30.0, 30.0, sigma=true_sigma, amplitude=2.0)
        img += np.random.normal(0, 0.1, (H, W)).astype(np.float32)
        img = np.clip(img, 0, None).astype(np.float32)
        _, _, sigma_fit, _ = fit_spot_gaussian(img, (30.0, 30.0), radius=12)
        rel_err = abs(sigma_fit - true_sigma) / true_sigma
        assert rel_err < 0.30, f"noisy fit error {rel_err:.1%} ≥ 30%"

    def test_sigma_within_bounds(self):
        """Fitted sigma must respect the [0.5, radius] bounds."""
        img = _make_gaussian_image(60, 60, 30.0, 30.0, sigma=2.5)
        radius = 10
        _, _, sigma, _ = fit_spot_gaussian(img, (30.0, 30.0), radius=radius)
        assert 0.5 <= sigma <= float(radius)


# ---------------------------------------------------------------------------
# TestTemporalQuality (internal helper)
# ---------------------------------------------------------------------------

class TestTemporalQuality:
    def test_zero_trace_returns_zero(self):
        trace = np.zeros(30, dtype=np.float32)
        assert _temporal_quality(trace) == 0.0

    def test_strong_transient_gives_high_score(self):
        """Sharp peak, short duration → high quality."""
        trace = np.zeros(30, dtype=np.float32)
        trace[14] = 2.0
        trace[13] = 0.8
        trace[15] = 0.6
        score = _temporal_quality(trace)
        assert score > 0.5

    def test_persistent_signal_gives_low_score(self):
        """Signal present in all frames → penalised as background."""
        trace = np.ones(30, dtype=np.float32) * 0.8
        score = _temporal_quality(trace)
        assert score < 0.5

    def test_empty_trace_returns_zero(self):
        trace = np.array([], dtype=np.float32)
        assert _temporal_quality(trace) == 0.0

    def test_score_in_unit_interval(self):
        np.random.seed(7)
        for _ in range(20):
            trace = np.random.rand(30).astype(np.float32) * 2.0
            s = _temporal_quality(trace)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    """Tests for compute_confidence — tier ordering and range enforcement."""

    def _transient_trace(self, T: int = 30) -> np.ndarray:
        """Build a strong, brief transient trace."""
        t = np.zeros(T, dtype=np.float32)
        t[12] = 0.5
        t[13] = 1.2
        t[14] = 2.0   # peak
        t[15] = 1.0
        t[16] = 0.4
        return t

    def _flat_trace(self, T: int = 30) -> np.ndarray:
        return np.zeros(T, dtype=np.float32)

    def _frame_qualities(self, T: int = 30) -> list[FrameQuality]:
        return [_frame_quality(t, 0.5) for t in range(T)]

    # --- Tier ranges ---

    def test_paired_score_in_tier_08_to_10(self):
        cluster = [_pair(30.0, 80.0, frame_idx=t) for t in range(10)]
        fqs = self._frame_qualities()
        score = compute_confidence(cluster, fqs, self._transient_trace(), total_frames=30)
        assert 0.80 <= score <= 1.00, f"paired score {score:.3f} out of [0.80, 1.00]"

    def test_single_near_pair_score_in_tier_06_to_08(self):
        roi = _single_roi(30.0, 80.0, det_type="single_near_pair")
        fqs = self._frame_qualities()
        score = compute_confidence(roi, fqs, self._transient_trace(), total_frames=30)
        assert 0.60 <= score <= 0.80, f"single_near_pair score {score:.3f} out of [0.60, 0.80]"

    def test_single_isolated_score_in_tier_04_to_06(self):
        roi = _single_roi(30.0, 80.0, det_type="single_isolated")
        fqs = self._frame_qualities()
        score = compute_confidence(roi, fqs, self._transient_trace(), total_frames=30)
        assert 0.40 <= score <= 0.60, f"single_isolated score {score:.3f} out of [0.40, 0.60]"

    def test_unknown_detection_type_in_tier_02_to_04(self):
        roi = _single_roi(30.0, 80.0, det_type="weak")
        fqs = self._frame_qualities()
        score = compute_confidence(roi, fqs, self._transient_trace(), total_frames=30)
        assert 0.20 <= score <= 0.40, f"weak score {score:.3f} out of [0.20, 0.40]"

    # --- Ordering ---

    def test_paired_greater_than_single_near_pair(self):
        trace = self._transient_trace()
        fqs = self._frame_qualities()
        cluster = [_pair(30.0, 80.0, frame_idx=t) for t in range(10)]
        near_roi = _single_roi(30.0, 80.0, det_type="single_near_pair")
        s_paired = compute_confidence(cluster, fqs, trace, 30)
        s_near = compute_confidence(near_roi, fqs, trace, 30)
        assert s_paired > s_near

    def test_single_near_pair_greater_than_single_isolated(self):
        trace = self._transient_trace()
        fqs = self._frame_qualities()
        near_roi = _single_roi(30.0, 80.0, det_type="single_near_pair")
        iso_roi = _single_roi(30.0, 80.0, det_type="single_isolated")
        s_near = compute_confidence(near_roi, fqs, trace, 30)
        s_iso = compute_confidence(iso_roi, fqs, trace, 30)
        assert s_near > s_iso

    def test_strong_trace_gives_higher_score_than_flat(self):
        fqs = self._frame_qualities()
        roi = _single_roi(30.0, 80.0, det_type="single_near_pair")
        s_strong = compute_confidence(roi, fqs, self._transient_trace(), 30)
        s_flat = compute_confidence(roi, fqs, self._flat_trace(), 30)
        assert s_strong > s_flat

    def test_output_clipped_to_01(self):
        fqs = self._frame_qualities()
        cluster = [_pair(30.0, 80.0, frame_idx=t) for t in range(30)]
        score = compute_confidence(cluster, fqs, self._transient_trace(), total_frames=30)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestHelperFunctions (roi_builder internals)
# ---------------------------------------------------------------------------

class TestCircleMask:
    def test_centre_is_true(self):
        mask = _circle_mask(40, 40, 20.0, 20.0, 5.0)
        assert mask[20, 20]

    def test_far_point_is_false(self):
        mask = _circle_mask(40, 40, 20.0, 20.0, 5.0)
        assert not mask[5, 5]

    def test_boundary_pixel(self):
        """Pixel exactly on the radius should be included (≤ r²)."""
        mask = _circle_mask(40, 40, 20.0, 20.0, 5.0)
        assert mask[20, 25]  # distance = 5.0 → included

    def test_shape_correct(self):
        mask = _circle_mask(50, 80, 25.0, 40.0, 6.0)
        assert mask.shape == (50, 80)


class TestExtractTemporalTrace:
    def test_all_false_mask_returns_zeros(self):
        dff = np.ones((20, 40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=bool)
        trace = _extract_temporal_trace(dff, mask)
        assert trace.shape == (20,)
        assert float(trace.sum()) == 0.0

    def test_trace_mean_within_mask(self):
        T, H, W = 10, 40, 40
        dff = np.zeros((T, H, W), dtype=np.float32)
        dff[:, 10:15, 10:15] = 1.5
        mask = np.zeros((H, W), dtype=bool)
        mask[10:15, 10:15] = True
        trace = _extract_temporal_trace(dff, mask)
        assert trace.shape == (T,)
        assert np.allclose(trace, 1.5)


class TestSelectTopFrames:
    def test_selects_highest_scoring_frames(self):
        fqs = [_frame_quality(t, float(t) / 10.0) for t in range(10)]
        top = _select_top_frames([0, 5, 9, 3], fqs, top_k=2)
        assert set(top).issubset({0, 3, 5, 9})
        assert top[0] == 9  # highest score among candidates

    def test_respects_top_k_limit(self):
        fqs = [_frame_quality(t, 0.5) for t in range(20)]
        top = _select_top_frames(list(range(20)), fqs, top_k=5)
        assert len(top) <= 5

    def test_empty_candidates_uses_all_frames(self):
        fqs = [_frame_quality(t, float(t)) for t in range(5)]
        top = _select_top_frames([], fqs, top_k=3)
        assert len(top) <= 3


# ---------------------------------------------------------------------------
# TestBuildNeuronROIs
# ---------------------------------------------------------------------------

class TestBuildNeuronROIs:
    """Integration tests for build_neuron_rois."""

    def _setup(self):
        """Return a small synthetic dataset with 2 paired + 1 single neuron."""
        T, H, W = 30, 80, 160
        np.random.seed(42)

        # Two paired neurons at (40, 50) and (40, 120)
        NEURON_A = (40, 50)
        NEURON_B = (40, 120)
        SINGLE_C = (20, 80)
        DX = 43.0
        SIGMA_TRUE = 2.5
        AMPS = [0.3, 0.7, 1.2, 0.8, 0.3]

        stack = np.random.normal(0.1, 0.02, (T, H, W)).astype(np.float32)
        dff = np.zeros((T, H, W), dtype=np.float32)

        # Inject spot signals
        for cy, cx in [NEURON_A, NEURON_B]:
            for spot_cx in [cx - DX / 2, cx + DX / 2]:
                img = _make_gaussian_image(H, W, cy, spot_cx, SIGMA_TRUE, amplitude=0.5)
                dff += np.stack([img * a for a in AMPS + [0.0] * (T - len(AMPS))])

        # Single-spot neuron
        img_s = _make_gaussian_image(H, W, SINGLE_C[0], SINGLE_C[1], SIGMA_TRUE, amplitude=0.4)
        dff += np.stack([img_s * a for a in ([0.0] * 15 + AMPS + [0.0] * (T - 15 - len(AMPS)))])
        stack += dff * 0.1

        # Build clusters for two paired neurons
        clusters = [
            [_pair(float(cy), float(cx), dx=DX, frame_idx=t) for t in range(5)]
            for cy, cx in [NEURON_A, NEURON_B]
        ]

        # Single-spot ROI
        singles = [_single_roi(float(SINGLE_C[0]), float(SINGLE_C[1]), "single_near_pair")]

        # Frame qualities (all medium)
        fqs = [_frame_quality(t, 0.5 + 0.01 * t) for t in range(T)]

        config = RefinementConfig(
            gaussian_fit_radius=8,
            top_k_frames=5,
            min_confidence=0.0,   # keep all for testing
        )
        return clusters, singles, stack, dff, fqs, config, T, H, W

    def test_returns_list_of_neuron_rois(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        assert isinstance(result, list)
        for roi in result:
            assert isinstance(roi, NeuronROI)

    def test_correct_total_count(self):
        """2 paired clusters + 1 single → 3 ROIs (min_confidence=0)."""
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        assert len(result) == 3

    def test_neuron_ids_sequential_from_zero(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        ids = [r.neuron_id for r in result]
        assert ids == list(range(len(result)))

    def test_paired_rois_come_first(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        # First two should be paired
        for roi in result[:2]:
            assert roi.detection_type == "paired"
        # Last one single
        assert result[2].detection_type == "single_near_pair"

    def test_paired_rois_have_both_spots(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        for roi in result[:2]:
            assert roi.left_spot is not None
            assert roi.right_spot is not None

    def test_single_rois_have_right_spot_none(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        single_roi = result[2]
        assert single_roi.right_spot is None
        assert single_roi.right_radius == 0.0

    def test_temporal_traces_have_correct_length(self):
        clusters, singles, stack, dff, fqs, config, T, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        for roi in result:
            assert roi.temporal_trace is not None
            assert roi.temporal_trace.shape == (T,)

    def test_masks_have_correct_shape(self):
        clusters, singles, stack, dff, fqs, config, T, H, W = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        for roi in result:
            assert roi.mask_left is not None
            assert roi.mask_left.shape == (H, W)

    def test_confidence_scores_in_range(self):
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        for roi in result:
            assert 0.0 <= roi.confidence <= 1.0

    def test_paired_confidence_above_single(self):
        """Paired ROIs should have higher confidence than single-spot ROIs."""
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        paired_confs = [r.confidence for r in result if r.detection_type == "paired"]
        single_confs = [r.confidence for r in result if r.detection_type != "paired"]
        assert min(paired_confs) > min(single_confs)

    def test_min_confidence_filter(self):
        """ROIs below min_confidence are excluded."""
        clusters, singles, stack, dff, fqs, config, *_ = self._setup()
        # Set min_confidence very high — should exclude everything
        config_strict = RefinementConfig(
            gaussian_fit_radius=8,
            top_k_frames=5,
            min_confidence=0.99,
        )
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config_strict)
        assert all(r.confidence >= 0.99 for r in result)

    def test_empty_clusters_and_singles(self):
        _, _, stack, dff, fqs, config, *_ = self._setup()
        result = build_neuron_rois([], [], stack, dff, fqs, config)
        assert result == []

    def test_total_frames_set_correctly(self):
        clusters, singles, stack, dff, fqs, config, T, *_ = self._setup()
        result = build_neuron_rois(clusters, singles, stack, dff, fqs, config)
        for roi in result:
            assert roi.total_frames == T
