"""Tests for src/bessel_seg/fusion/ — frame_quality, spatial_cluster, temporal_validate.

Synthetic data strategy
-----------------------
* All stacks are small (20–30 frames, 60×120 px) for speed.
* Spot / SpotPair helpers keep construction concise.
* np.random.seed(42) in every test that involves randomness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bessel_seg.config import ClusteringConfig, TemporalValidationConfig
from bessel_seg.data_types import FrameQuality, Spot, SpotPair
from bessel_seg.fusion.frame_quality import compute_frame_quality
from bessel_seg.fusion.spatial_cluster import cluster_neuron_detections
from bessel_seg.fusion.temporal_validate import (
    _cluster_centre,
    _extract_trace,
    _find_events,
    validate_calcium_dynamics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spot(y: float, x: float, sigma: float = 2.5, intensity: float = 1.0,
          frame_idx: int | None = None) -> Spot:
    return Spot(y=y, x=x, sigma=sigma, intensity=intensity, frame_idx=frame_idx)


def _pair(cy: float, cx: float, dx: float = 43.0, frame_idx: int | None = None,
          pair_cost: float = 0.1) -> SpotPair:
    left = _spot(cy, cx - dx / 2, frame_idx=frame_idx)
    right = _spot(cy, cx + dx / 2, frame_idx=frame_idx)
    return SpotPair(
        left=left,
        right=right,
        pair_distance=dx,
        horizontal_distance=dx,
        vertical_offset=0.0,
        intensity_ratio=1.0,
        scale_similarity=1.0,
        pair_cost=pair_cost,
    )


def _blank_stack(T: int = 20, H: int = 60, W: int = 120,
                 fill: float = 0.0) -> np.ndarray:
    return np.full((T, H, W), fill, dtype=np.float32)


def _dff_with_transient(T: int, H: int, W: int,
                        cy: int, cx: int,
                        start: int, amps: list[float],
                        radius: int = 4) -> np.ndarray:
    """Return a dff stack with a Gaussian transient at (cy, cx)."""
    dff = np.zeros((T, H, W), dtype=np.float32)
    ys, xs = np.ogrid[:H, :W]
    spot_mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    for i, amp in enumerate(amps):
        t = start + i
        if 0 <= t < T:
            dff[t][spot_mask] = amp
    return dff


# ---------------------------------------------------------------------------
# TestComputeFrameQuality
# ---------------------------------------------------------------------------

class TestComputeFrameQuality:
    """Tests for compute_frame_quality."""

    def test_returns_one_quality_per_frame(self):
        T, H, W = 10, 60, 120
        stack = _blank_stack(T, H, W)
        result = compute_frame_quality(stack, {}, {})
        assert len(result) == T

    def test_frame_idx_ordering(self):
        T = 5
        stack = _blank_stack(T)
        result = compute_frame_quality(stack, {}, {})
        assert [q.frame_idx for q in result] == list(range(T))

    def test_returns_framequality_objects(self):
        stack = _blank_stack(3)
        result = compute_frame_quality(stack, {}, {})
        for q in result:
            assert isinstance(q, FrameQuality)

    def test_overall_score_in_unit_interval(self):
        T, H, W = 20, 60, 120
        np.random.seed(42)
        stack = np.random.rand(T, H, W).astype(np.float32)
        spots_per_frame = {t: [_spot(30, 60)] for t in range(T)}
        pairs_per_frame = {t: [_pair(30, 60)] for t in range(0, T, 2)}
        result = compute_frame_quality(stack, spots_per_frame, pairs_per_frame)
        for q in result:
            assert 0.0 <= q.overall_score <= 1.0, f"out-of-range score at frame {q.frame_idx}"

    def test_pair_rate_zero_when_no_spots(self):
        stack = _blank_stack(5)
        result = compute_frame_quality(stack, {}, {})
        # No spots → pair_rate = 0 / max(0,1) = 0
        for q in result:
            assert q.pair_rate == 0.0

    def test_pair_rate_one_when_all_spots_paired(self):
        T = 5
        stack = _blank_stack(T)
        spots = {t: [_spot(30, 40), _spot(30, 83)] for t in range(T)}
        pairs = {t: [_pair(30, 61.5, dx=43.0)] for t in range(T)}
        result = compute_frame_quality(stack, spots, pairs)
        for q in result:
            assert math.isclose(q.pair_rate, 0.5, abs_tol=0.01)  # 1 pair / 2 spots

    def test_pair_rate_below_one_with_orphans(self):
        T = 4
        stack = _blank_stack(T)
        # 4 spots, 1 pair
        spots = {t: [_spot(10, 10), _spot(10, 53), _spot(40, 80), _spot(50, 90)]
                 for t in range(T)}
        pairs = {t: [_pair(10, 31.5, dx=43.0)] for t in range(T)}
        result = compute_frame_quality(stack, spots, pairs)
        for q in result:
            assert q.pair_rate < 1.0

    def test_snr_higher_for_bright_spots(self):
        T, H, W = 10, 60, 120
        # Dim stack
        dim_stack = _blank_stack(T, H, W, fill=0.01)
        # Bright spots injected in bright_stack
        bright_stack = _blank_stack(T, H, W, fill=0.01)
        cy, cx, r = 30, 60, 4
        bright_stack[:, cy - r:cy + r + 1, cx - r:cx + r + 1] = 1.0
        spots = {t: [_spot(cy, cx)] for t in range(T)}
        dim_q = compute_frame_quality(dim_stack, spots, {})
        bright_q = compute_frame_quality(bright_stack, spots, {})
        mean_dim_snr = np.mean([q.snr for q in dim_q])
        mean_bright_snr = np.mean([q.snr for q in bright_q])
        assert mean_bright_snr > mean_dim_snr

    def test_sharpness_higher_for_sharp_frame(self):
        T, H, W = 10, 60, 120
        # Uniform = zero Laplacian → low sharpness
        uniform_stack = _blank_stack(T, H, W, fill=0.5)
        # Checkerboard = high Laplacian
        checker = np.indices((H, W)).sum(axis=0) % 2
        sharp_stack = np.broadcast_to(checker[np.newaxis], (T, H, W)).astype(np.float32)
        q_uniform = compute_frame_quality(uniform_stack, {}, {})
        q_sharp = compute_frame_quality(sharp_stack, {}, {})
        mean_unif = np.mean([q.sharpness for q in q_uniform])
        mean_sharp = np.mean([q.sharpness for q in q_sharp])
        assert mean_sharp > mean_unif

    def test_empty_stack_does_not_raise(self):
        stack = _blank_stack(0)
        result = compute_frame_quality(stack, {}, {})
        assert result == []

    def test_single_frame_stack(self):
        stack = _blank_stack(1)
        result = compute_frame_quality(stack, {}, {})
        assert len(result) == 1
        assert result[0].frame_idx == 0

    def test_snr_zero_when_no_spots(self):
        stack = _blank_stack(5)
        result = compute_frame_quality(stack, {}, {})
        for q in result:
            assert q.snr == 0.0


# ---------------------------------------------------------------------------
# TestClusterNeuronDetections
# ---------------------------------------------------------------------------

class TestClusterNeuronDetections:
    """Tests for cluster_neuron_detections."""

    def _default_config(self, eps: float = 8.0, min_samples: int = 3) -> ClusteringConfig:
        return ClusteringConfig(eps=eps, min_samples=min_samples)

    def test_empty_pairs_returns_empty(self):
        config = self._default_config()
        result = cluster_neuron_detections([], config)
        assert result == []

    def test_two_spatially_separated_neurons(self):
        """Pairs far apart → two clusters."""
        config = self._default_config()
        neuron_a = [_pair(30.0 + np.random.normal(0, 1), 60.0, frame_idx=t)
                    for t in range(5)]
        neuron_b = [_pair(30.0 + np.random.normal(0, 1), 60.0 + 100, frame_idx=t)
                    for t in range(5)]
        np.random.seed(42)
        neuron_a = [_pair(30.0, 60.0, frame_idx=t) for t in range(5)]
        neuron_b = [_pair(30.0, 160.0, frame_idx=t) for t in range(5)]
        result = cluster_neuron_detections(neuron_a + neuron_b, config)
        assert len(result) == 2

    def test_cluster_sizes_sum_to_input_without_noise(self):
        """All points tightly grouped → no noise, cluster sizes = input size."""
        config = self._default_config(eps=10.0, min_samples=3)
        np.random.seed(42)
        pairs = [_pair(30.0 + np.random.normal(0, 1), 60.0, frame_idx=t)
                 for t in range(10)]
        result = cluster_neuron_detections(pairs, config)
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_noise_points_discarded(self):
        """Isolated single pair below min_samples → discarded."""
        config = self._default_config(min_samples=4)
        # 5 tight pairs (will cluster) + 1 isolated pair (noise)
        main = [_pair(30.0, 60.0, frame_idx=t) for t in range(5)]
        isolated = [_pair(30.0, 200.0, frame_idx=0)]   # only 1 → below min_samples=4
        result = cluster_neuron_detections(main + isolated, config)
        # Only the main cluster is returned; isolated is noise
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_clusters_sorted_by_descending_size(self):
        """Larger cluster first."""
        config = self._default_config()
        big = [_pair(30.0, 60.0, frame_idx=t) for t in range(8)]
        small = [_pair(30.0, 160.0, frame_idx=t) for t in range(3)]
        result = cluster_neuron_detections(big + small, config)
        assert len(result) == 2
        assert len(result[0]) >= len(result[1])

    def test_three_neurons_separated(self):
        config = self._default_config()
        a = [_pair(30.0, 50.0, frame_idx=t) for t in range(4)]
        b = [_pair(30.0, 150.0, frame_idx=t) for t in range(4)]
        c = [_pair(30.0, 250.0, frame_idx=t) for t in range(4)]
        result = cluster_neuron_detections(a + b + c, config)
        assert len(result) == 3

    def test_frame_index_ignored(self):
        """Same spatial position across many frames → single cluster."""
        config = self._default_config()
        pairs = [_pair(40.0, 80.0, frame_idx=t) for t in range(20)]
        result = cluster_neuron_detections(pairs, config)
        assert len(result) == 1

    def test_eps_controls_merge(self):
        """With small eps, nearby neurons stay separate; large eps merges them."""
        # Two neurons 15 px apart
        a = [_pair(30.0, 60.0, frame_idx=t) for t in range(5)]
        b = [_pair(30.0, 75.0, frame_idx=t) for t in range(5)]
        config_tight = self._default_config(eps=5.0)
        config_loose = self._default_config(eps=20.0)
        result_tight = cluster_neuron_detections(a + b, config_tight)
        result_loose = cluster_neuron_detections(a + b, config_loose)
        # Tight eps → 2 clusters (or fewer if noise); loose → 1 cluster
        assert len(result_loose) <= len(result_tight)

    def test_output_is_list_of_lists_of_spotpairs(self):
        config = self._default_config()
        pairs = [_pair(30.0, 60.0, frame_idx=t) for t in range(5)]
        result = cluster_neuron_detections(pairs, config)
        for cluster in result:
            assert isinstance(cluster, list)
            for p in cluster:
                assert isinstance(p, SpotPair)


# ---------------------------------------------------------------------------
# TestExtractTrace
# ---------------------------------------------------------------------------

class TestExtractTrace:
    """Tests for the internal _extract_trace helper."""

    def test_trace_length_equals_T(self):
        T, H, W = 20, 60, 120
        dff = _blank_stack(T, H, W)
        trace = _extract_trace(dff, 30.0, 60.0)
        assert trace.shape == (T,)

    def test_trace_dtype_float32(self):
        dff = _blank_stack(10)
        trace = _extract_trace(dff, 30.0, 60.0)
        assert trace.dtype == np.float32

    def test_trace_captures_injected_signal(self):
        T, H, W = 20, 60, 120
        # Inject spot with radius 4px; ROI extraction uses radius 6px, so the
        # mean is diluted (~0.44×).  Use large amplitudes so the diluted peak
        # is still well above background and shows a clear shape.
        dff = _dff_with_transient(T, H, W, cy=30, cx=60, start=5,
                                  amps=[0.6, 1.6, 2.0, 1.2, 0.4])
        trace = _extract_trace(dff, 30.0, 60.0)
        assert trace.max() > 0.5
        # Peak at frame 7 (start=5, peak is 3rd amp → frame 5+2=7)
        assert trace[7] > trace[5]  # peak frame is brighter than start frame

    def test_trace_zero_outside_spot(self):
        T, H, W = 20, 60, 120
        dff = _dff_with_transient(T, H, W, cy=30, cx=60, start=5, amps=[1.0])
        # Extract at far away location
        trace = _extract_trace(dff, 5.0, 5.0)
        assert float(trace.max()) < 0.01

    def test_border_centre_does_not_raise(self):
        T, H, W = 10, 60, 120
        dff = _blank_stack(T, H, W)
        trace = _extract_trace(dff, 0.0, 0.0)
        assert trace.shape == (T,)


# ---------------------------------------------------------------------------
# TestFindEvents
# ---------------------------------------------------------------------------

class TestFindEvents:
    """Tests for the internal _find_events helper."""

    def _config(self, **kwargs) -> TemporalValidationConfig:
        defaults = dict(
            min_peak_dff=0.20,
            min_rise_frames=1,
            min_decay_frames=1,
            max_event_duration=10,
            max_active_fraction=0.5,
            event_window=7,
        )
        defaults.update(kwargs)
        return TemporalValidationConfig(**defaults)

    def test_single_transient_detected(self):
        trace = np.zeros(20, dtype=np.float32)
        trace[8:13] = np.array([0.1, 0.4, 1.0, 0.5, 0.15], dtype=np.float32)
        events = _find_events(trace, self._config())
        assert len(events) >= 1

    def test_flat_zero_trace_no_events(self):
        trace = np.zeros(20, dtype=np.float32)
        events = _find_events(trace, self._config())
        assert events == []

    def test_persistent_signal_rejected_by_duration(self):
        """Signal active for 15 frames > max_event_duration=10 → rejected."""
        trace = np.ones(20, dtype=np.float32) * 0.8
        events = _find_events(trace, self._config(max_event_duration=10))
        # The whole trace is one long run of 20 frames > 10 → no events
        assert events == []

    def test_peak_below_threshold_rejected(self):
        trace = np.zeros(20, dtype=np.float32)
        trace[5:9] = 0.10  # below min_peak_dff=0.20
        events = _find_events(trace, self._config())
        assert events == []

    def test_multiple_events_detected(self):
        trace = np.zeros(40, dtype=np.float32)
        # Event 1 at frames 5-9
        trace[5:10] = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=np.float32)
        # Event 2 at frames 25-29
        trace[25:30] = np.array([0.1, 0.4, 0.9, 0.4, 0.1], dtype=np.float32)
        events = _find_events(trace, self._config())
        assert len(events) >= 2

    def test_event_tuple_structure(self):
        trace = np.zeros(20, dtype=np.float32)
        trace[5:10] = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=np.float32)
        events = _find_events(trace, self._config())
        assert len(events) >= 1
        start, end, peak = events[0]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(peak, float)
        assert start <= end
        assert peak >= 0.20


# ---------------------------------------------------------------------------
# TestValidateCalciumDynamics
# ---------------------------------------------------------------------------

class TestValidateCalciumDynamics:
    """Tests for validate_calcium_dynamics."""

    def _config(self, **kwargs) -> TemporalValidationConfig:
        defaults = dict(
            min_peak_dff=0.20,
            min_rise_frames=1,
            min_decay_frames=1,
            max_event_duration=10,
            max_active_fraction=0.5,
            event_window=7,
        )
        defaults.update(kwargs)
        return TemporalValidationConfig(**defaults)

    def test_empty_clusters_returns_empty(self):
        dff = _blank_stack(20)
        result = validate_calcium_dynamics([], dff, self._config())
        assert result == []

    def test_genuine_transient_accepted(self):
        """Cluster at location with a genuine calcium transient → kept."""
        T, H, W = 30, 60, 120
        cy, cx = 30, 60
        # Rise at 8, peak at 10, decay at 12 → 5-frame event
        amps = [0.15, 0.4, 0.9, 1.2, 0.8, 0.35, 0.1]
        dff = _dff_with_transient(T, H, W, cy, cx, start=8, amps=amps)
        cluster = [_pair(float(cy), float(cx), frame_idx=t) for t in range(5)]
        result = validate_calcium_dynamics([cluster], dff, self._config())
        assert len(result) == 1

    def test_persistent_signal_rejected(self):
        """Cluster at location with constant bright signal → rejected."""
        T, H, W = 30, 60, 120
        cy, cx = 30, 60
        # Constant 1.0 across all frames → active_fraction=1.0 > 0.5
        dff = np.ones((T, H, W), dtype=np.float32)
        cluster = [_pair(float(cy), float(cx), frame_idx=t) for t in range(5)]
        result = validate_calcium_dynamics([cluster], dff, self._config())
        assert len(result) == 0

    def test_low_signal_rejected(self):
        """Cluster where dff peak < 0.20 everywhere → rejected."""
        T, H, W = 30, 60, 120
        dff = np.full((T, H, W), 0.05, dtype=np.float32)  # below threshold
        cluster = [_pair(30.0, 60.0, frame_idx=t) for t in range(5)]
        result = validate_calcium_dynamics([cluster], dff, self._config())
        assert len(result) == 0

    def test_multiple_clusters_independently_assessed(self):
        """Two clusters: one with transient, one persistent → only one kept."""
        T, H, W = 30, 60, 120
        cy_a, cx_a = 30, 50
        cy_b, cx_b = 30, 100

        dff_a = _dff_with_transient(T, H, W, cy_a, cx_a, start=8,
                                    amps=[0.2, 0.5, 1.0, 0.5, 0.2])
        # Make cluster B persistent
        dff = dff_a.copy()
        dff[:, cy_b - 5:cy_b + 6, cx_b - 5:cx_b + 6] = 1.0

        cluster_a = [_pair(float(cy_a), float(cx_a), frame_idx=t) for t in range(5)]
        cluster_b = [_pair(float(cy_b), float(cx_b), frame_idx=t) for t in range(5)]

        result = validate_calcium_dynamics([cluster_a, cluster_b], dff, self._config())
        assert len(result) == 1

    def test_all_clusters_accepted_when_all_have_transients(self):
        T, H, W = 30, 60, 120
        amps = [0.15, 0.4, 1.0, 0.5, 0.1]
        dff = _dff_with_transient(T, H, W, 20, 40, start=5, amps=amps)
        dff += _dff_with_transient(T, H, W, 20, 90, start=15, amps=amps)

        c1 = [_pair(20.0, 40.0, frame_idx=t) for t in range(4)]
        c2 = [_pair(20.0, 90.0, frame_idx=t) for t in range(4)]
        result = validate_calcium_dynamics([c1, c2], dff, self._config())
        assert len(result) == 2

    def test_cluster_centre_computed_from_pair_midpoints(self):
        """Verify _cluster_centre returns mean of pair.center tuples."""
        p1 = _pair(20.0, 60.0)
        p2 = _pair(30.0, 60.0)
        cy, cx = _cluster_centre([p1, p2])
        assert math.isclose(cy, 25.0, abs_tol=0.5)
        assert math.isclose(cx, 60.0, abs_tol=0.5)

    def test_output_is_subset_of_input(self):
        T, H, W = 30, 60, 120
        amps = [0.2, 0.5, 1.0, 0.5, 0.15]
        dff = _dff_with_transient(T, H, W, 30, 60, start=5, amps=amps)
        clusters = [
            [_pair(30.0, 60.0, frame_idx=t) for t in range(5)],
            [_pair(30.0, 60.0, frame_idx=t) for t in range(5)],
        ]
        result = validate_calcium_dynamics(clusters, dff, self._config())
        # Every returned cluster must be one of the input clusters
        for r in result:
            assert r in clusters

    def test_order_preserved(self):
        """Validated clusters appear in the same relative order as input."""
        T, H, W = 30, 60, 120
        amps = [0.15, 0.5, 1.0, 0.5, 0.1]
        dff = _dff_with_transient(T, H, W, 20, 50, start=5, amps=amps)
        dff += _dff_with_transient(T, H, W, 20, 100, start=15, amps=amps)

        c1 = [_pair(20.0, 50.0, frame_idx=t) for t in range(4)]
        c2 = [_pair(20.0, 100.0, frame_idx=t) for t in range(4)]
        result = validate_calcium_dynamics([c1, c2], dff, self._config())
        if len(result) == 2:
            assert result[0] is c1
            assert result[1] is c2
