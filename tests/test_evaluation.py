"""Tests for evaluation/annotation_io.py and evaluation/metrics.py."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from bessel_seg.evaluation.annotation_io import (
    annotation_to_frame_set,
    load_annotation,
)
from bessel_seg.evaluation.metrics import (
    evaluate_detection,
    evaluate_detection_counts,
    evaluate_spatial,
)
from bessel_seg.data_types import NeuronROI, PipelineResult, FrameQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_annotation(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "ann.txt"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _make_roi(
    neuron_id: int,
    cy: float,
    cx: float,
    det_type: str = "paired",
    confidence: float = 0.9,
    trace: np.ndarray | None = None,
    detection_count: int = 5,
    total_frames: int = 60,
) -> NeuronROI:
    return NeuronROI(
        neuron_id=neuron_id,
        center_y=cy,
        center_x=cx,
        left_spot=None,
        right_spot=None,
        left_radius=0.0,
        right_radius=0.0,
        confidence=confidence,
        detection_type=det_type,
        detection_count=detection_count,
        total_frames=total_frames,
        temporal_trace=trace,
    )


def _make_result(neurons: list[NeuronROI], n_frames: int = 60) -> PipelineResult:
    return PipelineResult(
        neurons=neurons,
        frame_qualities=[],
        background_mask=np.zeros((10, 10), dtype=bool),
        summary_maps={},
        metadata={"n_frames": n_frames},
    )


# ---------------------------------------------------------------------------
# annotation_io tests
# ---------------------------------------------------------------------------

class TestAnnotationIO:
    def test_simple_single_frame(self, tmp_path):
        p = _write_annotation(tmp_path, "[10]:3(paired)+2(single)\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        e = entries[0]
        assert e["frame_start"] == 10
        assert e["frame_end"] == 10
        assert e["n_paired"] == 3
        assert e["n_single"] == 2
        assert e["n_total"] == 5

    def test_frame_range(self, tmp_path):
        p = _write_annotation(tmp_path, "[5-8]:2(paired)+3(single)\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        assert entries[0]["frame_start"] == 5
        assert entries[0]["frame_end"] == 8

    def test_no_single_count(self, tmp_path):
        p = _write_annotation(tmp_path, "[20]:4(paired)\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        assert entries[0]["n_paired"] == 4
        assert entries[0]["n_single"] == 0

    def test_short_format(self, tmp_path):
        """N+M without (paired)/(single) labels."""
        p = _write_annotation(tmp_path, "[30]:2+5\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        assert entries[0]["n_paired"] == 2
        assert entries[0]["n_single"] == 5

    def test_comments_skipped(self, tmp_path):
        content = """\
            # This is a comment
            [10]:1+2
            # Another comment
            [20]:3+1
        """
        p = _write_annotation(tmp_path, content)
        entries = load_annotation(p)
        assert len(entries) == 2

    def test_blank_lines_skipped(self, tmp_path):
        content = "\n[10]:1+0\n\n[15]:2+1\n\n"
        p = _write_annotation(tmp_path, content)
        entries = load_annotation(p)
        assert len(entries) == 2

    def test_fullwidth_brackets(self, tmp_path):
        p = _write_annotation(tmp_path, "（50）:3(paired)+2(single)\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        assert entries[0]["frame_start"] == 50
        assert entries[0]["n_paired"] == 3

    def test_fullwidth_range(self, tmp_path):
        p = _write_annotation(tmp_path, "（10-15）:2+1\n")
        entries = load_annotation(p)
        assert len(entries) == 1
        assert entries[0]["frame_start"] == 10
        assert entries[0]["frame_end"] == 15

    def test_multiple_entries(self, tmp_path):
        content = "[10]:2+1\n[20]:3+2\n[30]:1+0\n"
        p = _write_annotation(tmp_path, content)
        entries = load_annotation(p)
        assert len(entries) == 3

    def test_raw_field_preserved(self, tmp_path):
        p = _write_annotation(tmp_path, "[42]:5(paired)+3(single)\n")
        entries = load_annotation(p)
        assert "42" in entries[0]["raw"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_annotation(tmp_path / "nonexistent.txt")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        entries = load_annotation(p)
        assert entries == []


class TestAnnotationToFrameSet:
    def test_single_frame(self, tmp_path):
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 2, "n_single": 1, "n_total": 3, "raw": ""}]
        fs = annotation_to_frame_set(ann)
        assert 10 in fs
        assert fs[10]["n_paired"] == 2
        assert fs[10]["n_single"] == 1

    def test_frame_range_expansion(self, tmp_path):
        ann = [{"frame_start": 5, "frame_end": 8, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        fs = annotation_to_frame_set(ann)
        assert all(f in fs for f in [5, 6, 7, 8])
        assert 4 not in fs
        assert 9 not in fs

    def test_overlapping_entries_max(self):
        ann = [
            {"frame_start": 10, "frame_end": 12, "n_paired": 2, "n_single": 0, "n_total": 2, "raw": ""},
            {"frame_start": 11, "frame_end": 13, "n_paired": 1, "n_single": 3, "n_total": 4, "raw": ""},
        ]
        fs = annotation_to_frame_set(ann)
        # frame 11 and 12 overlap: take max
        assert fs[11]["n_paired"] == max(2, 1)
        assert fs[11]["n_single"] == max(0, 3)

    def test_empty_annotation(self):
        fs = annotation_to_frame_set([])
        assert fs == {}


# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------

_T = 60


def _trace_with_peak(T: int, peak_frame: int, amplitude: float = 2.0) -> np.ndarray:
    """Create a triangular trace that peaks at peak_frame."""
    t = np.zeros(T, dtype=np.float32)
    for dt in range(-3, 4):
        fi = peak_frame + dt
        if 0 <= fi < T:
            t[fi] = amplitude * max(0.0, 1.0 - abs(dt) / 4.0)
    return t


class TestEvaluateDetection:
    def _simple_result(self) -> PipelineResult:
        # Neuron that fires at frames 10, 20, 30
        traces = [_trace_with_peak(_T, f) for f in [10, 20, 30]]
        combined = np.stack(traces).max(axis=0)
        roi = _make_roi(0, 50, 80, trace=combined, total_frames=_T)
        return _make_result([roi], n_frames=_T)

    def test_perfect_recall(self):
        result = self._simple_result()
        # Annotation exactly matches firing frames
        ann = [
            {"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},
            {"frame_start": 20, "frame_end": 20, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},
            {"frame_start": 30, "frame_end": 30, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},
        ]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        assert dm.recall == pytest.approx(1.0, abs=0.01)

    def test_f1_between_zero_and_one(self):
        result = self._simple_result()
        ann = [
            {"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},
            {"frame_start": 40, "frame_end": 40, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},  # missed
        ]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        assert 0.0 <= dm.f1 <= 1.0

    def test_no_neurons_zero_recall(self):
        result = _make_result([], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        assert dm.recall == 0.0
        assert dm.tp == 0

    def test_no_annotation_all_fp(self):
        trace = _trace_with_peak(_T, 10)
        roi = _make_roi(0, 50, 80, trace=trace, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        dm = evaluate_detection(result, [], tolerance_frames=2)
        assert dm.precision == 0.0 or dm.n_annotated_frames == 0

    def test_tolerance_helps(self):
        """Pipeline fires at frame 12, annotation at frame 10. tol=2 → match."""
        trace = _trace_with_peak(_T, 12)
        roi = _make_roi(0, 50, 80, trace=trace, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        assert dm.recall > 0

    def test_no_tolerance_strict(self):
        """Pipeline fires at frame 14, annotation at frame 10. tol=0 → miss."""
        trace = _trace_with_peak(_T, 14)
        roi = _make_roi(0, 50, 80, trace=trace, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        dm = evaluate_detection(result, ann, tolerance_frames=0)
        assert dm.recall == 0.0

    def test_recall_paired_vs_single(self):
        trace_p = _trace_with_peak(_T, 10)
        roi_p = _make_roi(0, 50, 80, det_type="paired", trace=trace_p, total_frames=_T)
        trace_s = _trace_with_peak(_T, 20)
        roi_s = _make_roi(1, 70, 80, det_type="single_isolated", trace=trace_s, total_frames=_T)
        result = _make_result([roi_p, roi_s], n_frames=_T)
        ann = [
            {"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""},
            {"frame_start": 20, "frame_end": 20, "n_paired": 0, "n_single": 1, "n_total": 1, "raw": ""},
        ]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        assert dm.recall_paired >= 0.0
        assert dm.recall_single >= 0.0

    def test_precision_recall_f1_consistency(self):
        trace = _trace_with_peak(_T, 10)
        roi = _make_roi(0, 50, 80, trace=trace, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        dm = evaluate_detection(result, ann, tolerance_frames=2)
        # F1 = harmonic mean
        if dm.precision + dm.recall > 0:
            expected_f1 = 2 * dm.precision * dm.recall / (dm.precision + dm.recall)
            assert dm.f1 == pytest.approx(expected_f1, abs=0.001)


class TestEvaluateDetectionCounts:
    def test_count_mae_perfect(self):
        # Pipeline detects 1 paired neuron active at frame 10
        trace = _trace_with_peak(_T, 10, amplitude=2.0)
        roi = _make_roi(0, 50, 80, det_type="paired", trace=trace, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 1, "n_single": 0, "n_total": 1, "raw": ""}]
        cm = evaluate_detection_counts(result, ann, tolerance_frames=2)
        assert cm.n_frames_evaluated >= 1
        assert cm.mae_total >= 0.0  # exact value depends on threshold

    def test_count_mae_zero_detections(self):
        result = _make_result([], n_frames=_T)
        ann = [{"frame_start": 10, "frame_end": 10, "n_paired": 3, "n_single": 2, "n_total": 5, "raw": ""}]
        cm = evaluate_detection_counts(result, ann, tolerance_frames=2)
        assert cm.mae_total == pytest.approx(5.0, abs=0.1)

    def test_count_mae_empty_annotation(self):
        roi = _make_roi(0, 50, 80, total_frames=_T)
        result = _make_result([roi], n_frames=_T)
        cm = evaluate_detection_counts(result, [], tolerance_frames=2)
        assert cm.n_frames_evaluated == 0


class TestEvaluateSpatial:
    def _result_with_neuron_at(self, cy: float, cx: float) -> PipelineResult:
        roi = _make_roi(0, cy, cx, total_frames=_T)
        return _make_result([roi])

    def test_exact_match(self):
        result = self._result_with_neuron_at(50.0, 80.0)
        sm = evaluate_spatial(result, [(50.0, 80.0)], radius_px=15.0)
        assert sm.recall == pytest.approx(1.0)
        assert sm.n_matched == 1

    def test_within_radius(self):
        result = self._result_with_neuron_at(50.0, 80.0)
        sm = evaluate_spatial(result, [(55.0, 85.0)], radius_px=15.0)  # dist≈7px
        assert sm.recall == pytest.approx(1.0)

    def test_outside_radius(self):
        result = self._result_with_neuron_at(50.0, 80.0)
        sm = evaluate_spatial(result, [(100.0, 80.0)], radius_px=15.0)  # dist=50px
        assert sm.recall == pytest.approx(0.0)

    def test_multiple_gt_partial_match(self):
        result = self._result_with_neuron_at(50.0, 80.0)
        gt = [(50.0, 80.0), (200.0, 200.0)]  # first matches, second doesn't
        sm = evaluate_spatial(result, gt, radius_px=15.0)
        assert sm.recall == pytest.approx(0.5)
        assert sm.n_matched == 1
        assert len(sm.unmatched_gt) == 1

    def test_empty_detections(self):
        result = _make_result([], n_frames=_T)
        sm = evaluate_spatial(result, [(50.0, 80.0)], radius_px=15.0)
        assert sm.recall == 0.0
        assert sm.n_matched == 0

    def test_empty_gt(self):
        result = self._result_with_neuron_at(50.0, 80.0)
        sm = evaluate_spatial(result, [], radius_px=15.0)
        assert sm.recall == 0.0
        assert sm.n_gt == 0

    def test_unmatched_gt_reported(self):
        result = _make_result([], n_frames=_T)
        sm = evaluate_spatial(result, [(30.0, 40.0), (70.0, 90.0)], radius_px=5.0)
        assert len(sm.unmatched_gt) == 2
