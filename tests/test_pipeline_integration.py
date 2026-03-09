"""Integration test for the full Bessel beam segmentation pipeline."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bessel_seg.config import PipelineConfig
from bessel_seg.data_types import PipelineResult
from bessel_seg.pipeline import run_pipeline


_SIGMA_SPOT: float = 2.5
_PAIR_DX: float = 43.0
_T, _H, _W = 60, 100, 160


def _gaussian_footprint(H, W, cy, cx, sigma):
    Y, X = np.ogrid[:H, :W]
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2)).astype(np.float32)


def _add_events(stack, cy, cx, sigma, amplitude, starts, duration):
    T, H, W = stack.shape
    fp = _gaussian_footprint(H, W, cy, cx, sigma)
    for start in starts:
        for dt in range(duration):
            t = start + dt
            if t >= T:
                break
            half = max(duration / 2.0, 1.0)
            weight = min(dt + 1, duration - dt) / half
            stack[t] += amplitude * weight * fp


def make_synthetic_stack(seed=42):
    """(T=60, H=100, W=160): 2 paired + 2 single + 1 persistent bg blob."""
    np.random.seed(seed)
    T, H, W = _T, _H, _W

    bg = np.ones((H, W), dtype=np.float32) * 12.0
    bg += np.linspace(3.0, 0.0, W, dtype=np.float32)[np.newaxis, :]
    for row in range(0, H, 18):
        bg[row:row+2, :] += 4.0

    bleach = np.linspace(1.0, 0.75, T, dtype=np.float32)[:, np.newaxis, np.newaxis]
    stack = np.tile(bg[np.newaxis], (T, 1, 1)) * bleach
    stack += np.random.normal(0.0, 0.3, (T, H, W)).astype(np.float32)
    stack = np.clip(stack, 0.0, None).astype(np.float32)

    # Neuron A (paired): center (50, 80)
    _add_events(stack, 50.0, 80.0 - _PAIR_DX/2, _SIGMA_SPOT, 30.0, [5, 25, 45], 5)
    _add_events(stack, 50.0, 80.0 + _PAIR_DX/2, _SIGMA_SPOT, 30.0, [5, 25, 45], 5)

    # Neuron B (paired): center (30, 110)
    _add_events(stack, 30.0, 110.0 - _PAIR_DX/2, _SIGMA_SPOT, 30.0, [10, 30, 50], 5)
    _add_events(stack, 30.0, 110.0 + _PAIR_DX/2, _SIGMA_SPOT, 30.0, [10, 30, 50], 5)

    # Neuron C (single, isolated): center (75, 40)
    _add_events(stack, 75.0, 40.0, _SIGMA_SPOT, 28.0, [15, 35, 53], 5)

    # Neuron D (single, isolated): center (20, 40)
    _add_events(stack, 20.0, 40.0, _SIGMA_SPOT, 28.0, [8, 28, 48], 5)

    # Persistent blob at (60, 148) — always bright -> REJECTED
    stack += 20.0 * _gaussian_footprint(H, W, 60.0, 148.0, 3.0)[np.newaxis, :, :]

    return stack.astype(np.float32)


def _test_config():
    cfg = PipelineConfig()
    cfg.preprocessing.registration_method = "none"
    cfg.preprocessing.photobleach_correction = "linear"
    cfg.deepcad.enabled = False
    cfg.detection.threshold = 0.004
    cfg.detection.local_snr_threshold = 3.0
    cfg.detection.min_sigma = 1.0
    cfg.detection.max_sigma = 5.0
    cfg.detection.num_sigma = 6
    cfg.pairing.horizontal_dist_min = 25
    cfg.pairing.horizontal_dist_max = 60
    cfg.pairing.vertical_offset_max = 8
    cfg.clustering.eps = 10.0
    cfg.clustering.min_samples = 2
    cfg.temporal_validation.min_peak_dff = 0.15
    cfg.temporal_validation.max_active_fraction = 0.60
    cfg.temporal_validation.min_rise_frames = 1
    cfg.temporal_validation.min_decay_frames = 1
    cfg.refinement.min_confidence = 0.0
    cfg.refinement.top_k_frames = 6
    cfg.refinement.gaussian_fit_radius = 8
    return cfg


@pytest.fixture(scope="module")
def synthetic_stack():
    return make_synthetic_stack(seed=42)


@pytest.fixture(scope="module")
def pipeline_result(synthetic_stack, tmp_path_factory):
    import tifffile
    tmp = tmp_path_factory.mktemp("integration_out")
    stack_path = tmp / "synth.tif"
    tifffile.imwrite(str(stack_path), synthetic_stack)
    result = run_pipeline(str(stack_path), config=_test_config(),
                          output_dir=str(tmp / "results"))
    return result


class TestPipelineStructure:
    def test_returns_pipeline_result(self, pipeline_result):
        assert isinstance(pipeline_result, PipelineResult)

    def test_neurons_is_list(self, pipeline_result):
        assert isinstance(pipeline_result.neurons, list)

    def test_frame_qualities_is_list(self, pipeline_result):
        assert isinstance(pipeline_result.frame_qualities, list)

    def test_background_mask_shape(self, pipeline_result):
        assert pipeline_result.background_mask.shape == (_H, _W)
        assert pipeline_result.background_mask.dtype == bool

    def test_summary_maps_keys(self, pipeline_result):
        for k in ("sigma", "max_proj", "mean", "cv", "edi"):
            assert k in pipeline_result.summary_maps

    def test_summary_maps_shapes(self, pipeline_result):
        for k, v in pipeline_result.summary_maps.items():
            assert v.shape == (_H, _W), f"Map '{k}' has wrong shape {v.shape}"

    def test_metadata_keys(self, pipeline_result):
        for k in ("n_frames", "n_neurons_total", "pipeline_elapsed_s"):
            assert k in pipeline_result.metadata

    def test_metadata_frame_count(self, pipeline_result):
        assert pipeline_result.metadata["n_frames"] == _T


class TestNeuronROIStructure:
    def test_at_least_two_neurons(self, pipeline_result):
        assert len(pipeline_result.neurons) >= 2, (
            f"Expected >=2 neurons, got {len(pipeline_result.neurons)}"
        )

    def test_neuron_ids_sequential(self, pipeline_result):
        ids = [r.neuron_id for r in pipeline_result.neurons]
        assert ids == list(range(len(ids)))

    def test_confidence_in_unit_interval(self, pipeline_result):
        for roi in pipeline_result.neurons:
            assert 0.0 <= roi.confidence <= 1.0

    def test_detection_type_valid(self, pipeline_result):
        valid = {"paired", "single_near_pair", "single_isolated"}
        for roi in pipeline_result.neurons:
            assert roi.detection_type in valid

    def test_temporal_trace_shape(self, pipeline_result):
        for roi in pipeline_result.neurons:
            if roi.temporal_trace is not None:
                assert roi.temporal_trace.shape == (_T,)

    def test_mask_shape(self, pipeline_result):
        for roi in pipeline_result.neurons:
            if roi.mask_left is not None:
                assert roi.mask_left.shape == (_H, _W)
            if roi.mask_right is not None:
                assert roi.mask_right.shape == (_H, _W)

    def test_paired_has_both_spots(self, pipeline_result):
        for roi in pipeline_result.neurons:
            if roi.detection_type == "paired":
                assert roi.left_spot is not None
                assert roi.right_spot is not None

    def test_single_right_spot_none(self, pipeline_result):
        for roi in pipeline_result.neurons:
            if roi.detection_type in {"single_near_pair", "single_isolated"}:
                assert roi.right_spot is None

    def test_center_within_image(self, pipeline_result):
        for roi in pipeline_result.neurons:
            assert 0 <= roi.center_y < _H
            assert 0 <= roi.center_x < _W

    def test_total_frames_correct(self, pipeline_result):
        for roi in pipeline_result.neurons:
            assert roi.total_frames == _T


class TestConfidenceTiers:
    def test_paired_confidence_above_070(self, pipeline_result):
        paired = [r for r in pipeline_result.neurons if r.detection_type == "paired"]
        if not paired:
            pytest.skip("No paired neurons detected")
        for roi in paired:
            assert roi.confidence >= 0.70

    def test_mean_paired_above_mean_single(self, pipeline_result):
        paired = [r for r in pipeline_result.neurons if r.detection_type == "paired"]
        single = [r for r in pipeline_result.neurons if r.detection_type != "paired"]
        if not paired or not single:
            pytest.skip("Need both paired and single neurons")
        assert float(np.mean([r.confidence for r in paired])) > float(
            np.mean([r.confidence for r in single])
        )


class TestNeuronLocations:
    GT = {"A": (50.0, 80.0), "B": (30.0, 110.0),
          "C": (75.0, 40.0), "D": (20.0, 40.0)}
    RADIUS = 15.0

    def _nearest(self, neurons, cy, cx):
        if not neurons:
            return float("inf"), None
        dists = [math.hypot(r.center_y - cy, r.center_x - cx) for r in neurons]
        i = int(np.argmin(dists))
        return dists[i], neurons[i]

    def test_paired_a_detected(self, pipeline_result):
        paired = [r for r in pipeline_result.neurons if r.detection_type == "paired"]
        d, _ = self._nearest(paired, *self.GT["A"])
        assert d < self.RADIUS, f"Neuron A not found (nearest paired: {d:.1f} px)"

    def test_paired_b_detected(self, pipeline_result):
        paired = [r for r in pipeline_result.neurons if r.detection_type == "paired"]
        d, _ = self._nearest(paired, *self.GT["B"])
        assert d < self.RADIUS, f"Neuron B not found (nearest paired: {d:.1f} px)"

    def test_single_c_detected(self, pipeline_result):
        d, _ = self._nearest(pipeline_result.neurons, *self.GT["C"])
        assert d < self.RADIUS, f"Neuron C not found (nearest: {d:.1f} px)"

    def test_single_d_detected(self, pipeline_result):
        d, _ = self._nearest(pipeline_result.neurons, *self.GT["D"])
        assert d < self.RADIUS, f"Neuron D not found (nearest: {d:.1f} px)"

    def test_background_blob_rejected(self, pipeline_result):
        d, roi = self._nearest(pipeline_result.neurons, 60.0, 148.0)
        assert roi is None or d > 10.0, (
            f"Background blob (60,148) was not rejected (nearest: {d:.1f} px)"
        )


class TestSavedOutputFiles:
    @pytest.fixture(scope="class")
    def output_dir(self, tmp_path_factory, synthetic_stack):
        import tifffile
        tmp = tmp_path_factory.mktemp("out_files")
        stack_path = tmp / "s.tif"
        tifffile.imwrite(str(stack_path), synthetic_stack)
        out = tmp / "results"
        run_pipeline(str(stack_path), config=_test_config(), output_dir=str(out))
        return out

    def test_neurons_json_exists(self, output_dir):
        assert (output_dir / "neurons.json").exists()

    def test_neurons_json_parseable(self, output_dir):
        import json
        data = json.loads((output_dir / "neurons.json").read_text())
        assert isinstance(data, list)

    def test_roi_masks_tif_exists(self, output_dir):
        assert (output_dir / "roi_masks.tif").exists()

    def test_traces_csv_exists(self, output_dir):
        assert (output_dir / "traces.csv").exists()

    def test_traces_csv_rows(self, output_dir):
        import pandas as pd
        df = pd.read_csv(output_dir / "traces.csv")
        assert len(df) == _T

    def test_frame_quality_csv_exists(self, output_dir):
        assert (output_dir / "frame_quality.csv").exists()

    def test_summary_maps_npz_exists(self, output_dir):
        assert (output_dir / "summary_maps.npz").exists()

    def test_report_html_exists(self, output_dir):
        assert (output_dir / "report.html").exists()

    def test_report_html_content(self, output_dir):
        html = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "<html" in html.lower()


class TestFourNeuronsDetected:
    """Primary assertion: pipeline finds exactly 4 ground-truth neurons."""

    def test_exactly_four_neurons(self, pipeline_result):
        n = len(pipeline_result.neurons)
        summary = "\n".join(
            f"  id={r.neuron_id} type={r.detection_type} "
            f"cy={r.center_y:.1f} cx={r.center_x:.1f} conf={r.confidence:.3f}"
            for r in pipeline_result.neurons
        )
        assert n == 4, f"Expected 4 neurons (2 paired + 2 single), got {n}.\n{summary}"

    def test_two_paired_two_single(self, pipeline_result):
        n_p = sum(1 for r in pipeline_result.neurons if r.detection_type == "paired")
        n_s = len(pipeline_result.neurons) - n_p
        assert n_p == 2, f"Expected 2 paired neurons, got {n_p}"
        assert n_s == 2, f"Expected 2 single-spot neurons, got {n_s}"

    def test_paired_confidence_gte_080(self, pipeline_result):
        for roi in pipeline_result.neurons:
            if roi.detection_type == "paired":
                assert roi.confidence >= 0.80, (
                    f"Paired neuron {roi.neuron_id} confidence {roi.confidence:.3f} < 0.80"
                )

    def test_paired_confidence_beats_single(self, pipeline_result):
        paired = [r for r in pipeline_result.neurons if r.detection_type == "paired"]
        single = [r for r in pipeline_result.neurons if r.detection_type != "paired"]
        if not paired or not single:
            pytest.skip("Need both types")
        assert min(r.confidence for r in paired) > max(r.confidence for r in single) - 0.05
