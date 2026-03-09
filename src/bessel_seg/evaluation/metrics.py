"""Detection quality metrics for the Bessel beam neuron segmentation pipeline.

Metrics
-------
evaluate_detection(pipeline_result, annotation, tolerance_frames)
    Frame-level precision / recall / F1.  A pipeline frame is a "true positive"
    if the annotation has at least one annotated neuron within ±tolerance_frames
    of it, AND the pipeline detected at least one neuron in that frame.

evaluate_detection_counts(pipeline_result, annotation, tolerance_frames)
    For each annotated frame, compare pipeline neuron counts to annotation counts
    and compute mean absolute error in total / paired / single counts.

evaluate_spatial(pipeline_result, gt_positions, radius_px)
    Spatial recall: fraction of ground-truth neuron positions matched by a
    pipeline detection within radius_px pixels.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bessel_seg.data_types import NeuronROI, PipelineResult
from bessel_seg.evaluation.annotation_io import annotation_to_frame_set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    """Frame-level precision / recall / F1 for event detection."""
    precision: float
    recall: float
    f1: float
    tp: int          # True positives (annotated frames correctly detected)
    fp: int          # False positives (pipeline frames not in annotation ± tol)
    fn: int          # False negatives (annotated frames missed by pipeline)
    # Per-type recall
    recall_paired: float = 0.0
    recall_single: float = 0.0
    n_annotated_frames: int = 0
    n_pipeline_frames: int = 0


@dataclass
class CountMetrics:
    """Per-frame count accuracy."""
    mae_total: float    # Mean absolute error of total neuron count
    mae_paired: float
    mae_single: float
    n_frames_evaluated: int


@dataclass
class SpatialMetrics:
    """Spatial match of detected neurons to ground-truth positions."""
    recall: float
    n_gt: int
    n_matched: int
    unmatched_gt: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_pipeline_frames(
    result: PipelineResult,
    tolerance_frames: int,
) -> set[int]:
    """Return the set of frames where the pipeline detected ≥1 neuron.

    A neuron is "detected in frame t" if the neuron was actively firing in
    frame t, i.e. its temporal_trace[t] > 0 OR it was detected in a frame
    within ±tolerance_frames if we only have detection_count metadata.

    Since PipelineResult doesn't carry per-frame firing info directly, we use
    the temporal trace: frames where dff > 0.05 * peak are considered active.
    """
    active: set[int] = set()
    for roi in result.neurons:
        if roi.temporal_trace is not None and len(roi.temporal_trace) > 0:
            peak = float(roi.temporal_trace.max())
            threshold = max(0.05 * peak, 0.05)
            firing_frames = np.where(roi.temporal_trace > threshold)[0]
            active.update(int(f) for f in firing_frames)
    return active


def _expand_frames(frames: set[int], tolerance: int) -> set[int]:
    """Expand a set of frame indices by ±tolerance."""
    expanded: set[int] = set()
    for f in frames:
        for dt in range(-tolerance, tolerance + 1):
            expanded.add(f + dt)
    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_detection(
    pipeline_result: PipelineResult,
    annotation: list[dict],
    tolerance_frames: int = 2,
) -> DetectionMetrics:
    """Compute frame-level precision, recall, F1 against manual annotations.

    A pipeline detection frame matches an annotation entry if it falls within
    ±tolerance_frames of any annotated frame.

    A pipeline frame is considered "active" if ≥1 neuron has a temporal trace
    value exceeding 5% of its peak in that frame.

    Args:
        pipeline_result: Output of run_pipeline().
        annotation: List of dicts from load_annotation().
        tolerance_frames: Temporal tolerance window in frames.

    Returns:
        DetectionMetrics with precision, recall, F1 and breakdown.
    """
    frame_index = annotation_to_frame_set(annotation)
    annotated_frames = set(frame_index.keys())
    pipeline_frames = _active_pipeline_frames(pipeline_result, tolerance_frames)

    # Expand annotated frames by tolerance for matching
    annotated_expanded = _expand_frames(annotated_frames, tolerance_frames)

    # TP: pipeline frames that fall within expanded annotated window
    tp_frames = pipeline_frames & annotated_expanded
    tp = len(tp_frames)
    fp = len(pipeline_frames - annotated_expanded)

    # FN: annotated frames not covered by any pipeline frame (even with tolerance)
    pipeline_expanded = _expand_frames(pipeline_frames, tolerance_frames)
    fn_frames = annotated_frames - pipeline_expanded
    fn = len(fn_frames)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-type recall: fraction of paired-only / single-only annotated frames covered
    paired_frames = {f for f, d in frame_index.items() if d["n_paired"] > 0}
    single_frames = {f for f, d in frame_index.items() if d["n_single"] > 0}

    def _recall_subset(subset: set[int]) -> float:
        if not subset:
            return 0.0
        covered = subset & pipeline_expanded
        return len(covered) / len(subset)

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        recall_paired=_recall_subset(paired_frames),
        recall_single=_recall_subset(single_frames),
        n_annotated_frames=len(annotated_frames),
        n_pipeline_frames=len(pipeline_frames),
    )


def evaluate_detection_counts(
    pipeline_result: PipelineResult,
    annotation: list[dict],
    tolerance_frames: int = 2,
) -> CountMetrics:
    """Evaluate per-frame count accuracy vs. annotations.

    For each annotated frame, find the pipeline frame(s) within ±tolerance_frames
    and compare the counts.

    Args:
        pipeline_result: Output of run_pipeline().
        annotation: List of dicts from load_annotation().
        tolerance_frames: Temporal tolerance window.

    Returns:
        CountMetrics with mean absolute errors.
    """
    frame_index = annotation_to_frame_set(annotation)
    pipeline_frames = _active_pipeline_frames(pipeline_result, tolerance_frames)

    # Per-frame pipeline counts
    n_frames = (
        len(pipeline_result.frame_qualities)
        if pipeline_result.frame_qualities
        else pipeline_result.metadata.get("n_frames", 0)
    )

    # Build per-frame detection counts from temporal traces
    total_counts: dict[int, int] = {}
    paired_counts: dict[int, int] = {}
    single_counts: dict[int, int] = {}
    for roi in pipeline_result.neurons:
        if roi.temporal_trace is not None and len(roi.temporal_trace) > 0:
            peak = float(roi.temporal_trace.max())
            threshold = max(0.05 * peak, 0.05)
            for t in np.where(roi.temporal_trace > threshold)[0]:
                t = int(t)
                total_counts[t] = total_counts.get(t, 0) + 1
                if roi.detection_type == "paired":
                    paired_counts[t] = paired_counts.get(t, 0) + 1
                else:
                    single_counts[t] = single_counts.get(t, 0) + 1

    errs_total, errs_paired, errs_single = [], [], []
    for ann_frame, ann_data in frame_index.items():
        # Find best pipeline frame within tolerance
        candidate_frames = [
            f for f in range(ann_frame - tolerance_frames, ann_frame + tolerance_frames + 1)
            if f in pipeline_frames
        ]
        if candidate_frames:
            # Use the closest pipeline frame
            best = min(candidate_frames, key=lambda f: abs(f - ann_frame))
            p_total = total_counts.get(best, 0)
            p_paired = paired_counts.get(best, 0)
            p_single = single_counts.get(best, 0)
        else:
            p_total = p_paired = p_single = 0

        errs_total.append(abs(p_total - ann_data["n_total"]))
        errs_paired.append(abs(p_paired - ann_data["n_paired"]))
        errs_single.append(abs(p_single - ann_data["n_single"]))

    if not errs_total:
        return CountMetrics(mae_total=0.0, mae_paired=0.0, mae_single=0.0, n_frames_evaluated=0)

    return CountMetrics(
        mae_total=float(np.mean(errs_total)),
        mae_paired=float(np.mean(errs_paired)),
        mae_single=float(np.mean(errs_single)),
        n_frames_evaluated=len(errs_total),
    )


def evaluate_spatial(
    pipeline_result: PipelineResult,
    gt_positions: list[tuple[float, float]],
    radius_px: float = 15.0,
) -> SpatialMetrics:
    """Evaluate spatial recall against ground-truth neuron positions.

    Args:
        pipeline_result: Output of run_pipeline().
        gt_positions: List of (y, x) ground-truth neuron centre positions.
        radius_px: A GT position is "matched" if a pipeline detection is
                   within radius_px pixels of it.

    Returns:
        SpatialMetrics with recall and match details.
    """
    det_positions = [(r.center_y, r.center_x) for r in pipeline_result.neurons]
    matched = 0
    unmatched_gt: list[tuple[float, float]] = []

    for gt_y, gt_x in gt_positions:
        found = any(
            math.hypot(gt_y - dy, gt_x - dx) <= radius_px
            for dy, dx in det_positions
        )
        if found:
            matched += 1
        else:
            unmatched_gt.append((gt_y, gt_x))

    n_gt = len(gt_positions)
    recall = matched / n_gt if n_gt > 0 else 0.0
    return SpatialMetrics(recall=recall, n_gt=n_gt, n_matched=matched,
                          unmatched_gt=unmatched_gt)
