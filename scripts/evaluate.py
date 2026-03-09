#!/usr/bin/env python3
"""Evaluate pipeline results against manual ground-truth annotations.

Usage
-----
.. code-block:: bash

    python scripts/evaluate.py results/dataset_001/neurons.json data/annotations/record_001.txt
    python scripts/evaluate.py results/dataset_001/neurons.json data/annotations/record_001.txt \\
        --tol 2 --radius 15 --gt-positions 128.5,200.3 250.0,310.5

Arguments
---------
neurons_json
    Path to a ``neurons.json`` produced by the pipeline.

annotation_txt
    Path to a ground-truth annotation .txt file in the standard format.

--tol
    Temporal tolerance in frames for frame-level matching (default: 2).

--radius
    Spatial radius in pixels for GT position matching (default: 15).

--gt-positions
    Optional list of "y,x" ground-truth neuron positions for spatial recall
    evaluation.  Example: --gt-positions 128.5,200.3 250.0,310.5

--loglevel
    Python logging level (default: INFO).

Output
------
Prints a summary table of precision, recall, F1, and count MAE to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline detections against annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "neurons_json",
        type=str,
        help="Path to neurons.json from the pipeline.",
    )
    parser.add_argument(
        "annotation_txt",
        type=str,
        help="Path to annotation .txt file.",
    )
    parser.add_argument(
        "--tol",
        type=int,
        default=2,
        dest="tolerance_frames",
        help="Temporal tolerance in frames (default: 2).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=15.0,
        help="Spatial match radius in pixels (default: 15).",
    )
    parser.add_argument(
        "--gt-positions",
        nargs="+",
        default=None,
        metavar="Y,X",
        help=(
            "Ground-truth neuron positions as 'y,x' strings.  "
            "Example: --gt-positions 128.5,200.3 250.0,310.5"
        ),
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def _load_neurons_json(json_path: Path) -> list[dict]:
    """Load neurons.json and return a list of neuron dicts."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"neurons.json must be a JSON array, got {type(data)}")
    return data


def _make_fake_pipeline_result(neurons_data: list[dict], n_frames: int):
    """Build a minimal PipelineResult-like object from neurons.json for evaluation."""
    from bessel_seg.data_types import NeuronROI, PipelineResult
    import numpy as np

    neurons: list[NeuronROI] = []
    for d in neurons_data:
        # Reconstruct a minimal NeuronROI from JSON
        trace = None
        if d.get("temporal_trace") is not None:
            trace = np.array(d["temporal_trace"], dtype=np.float32)
        roi = NeuronROI(
            neuron_id=d.get("neuron_id", 0),
            center_y=float(d.get("center_y", 0)),
            center_x=float(d.get("center_x", 0)),
            left_spot=None,
            right_spot=None,
            left_radius=0.0,
            right_radius=0.0,
            confidence=float(d.get("confidence", 0)),
            detection_type=d.get("detection_type", "single_isolated"),
            detection_count=int(d.get("detection_count", 0)),
            total_frames=int(d.get("total_frames", n_frames)),
            temporal_trace=trace,
        )
        neurons.append(roi)

    return PipelineResult(
        neurons=neurons,
        frame_qualities=[],
        background_mask=np.zeros((1, 1), dtype=bool),
        summary_maps={},
        metadata={"n_frames": n_frames},
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from bessel_seg.evaluation.annotation_io import load_annotation
    from bessel_seg.evaluation.metrics import (
        evaluate_detection,
        evaluate_detection_counts,
        evaluate_spatial,
    )

    # ---- Load inputs ----
    neurons_json_path = Path(args.neurons_json)
    annotation_txt_path = Path(args.annotation_txt)

    if not neurons_json_path.exists():
        logger.error("neurons.json not found: %s", neurons_json_path)
        return 1
    if not annotation_txt_path.exists():
        logger.error("annotation file not found: %s", annotation_txt_path)
        return 1

    neurons_data = _load_neurons_json(neurons_json_path)
    annotation = load_annotation(annotation_txt_path)

    if not annotation:
        logger.warning("No annotation entries parsed from %s", annotation_txt_path)

    # Guess n_frames from neuron metadata or annotation
    n_frames = max(
        (d.get("total_frames", 0) for d in neurons_data),
        default=0,
    ) or max(
        (a["frame_end"] for a in annotation), default=300
    ) + 1

    pipeline_result = _make_fake_pipeline_result(neurons_data, n_frames)
    logger.info(
        "Loaded %d neurons, %d annotation entries, n_frames=%d",
        len(neurons_data), len(annotation), n_frames,
    )

    # ---- Frame-level metrics ----
    dm = evaluate_detection(pipeline_result, annotation, args.tolerance_frames)
    cm = evaluate_detection_counts(pipeline_result, annotation, args.tolerance_frames)

    print(f"\n{'=' * 56}")
    print(f"  Evaluation: {neurons_json_path.parent.name}")
    print(f"  Annotation: {annotation_txt_path.name}")
    print(f"  Tolerance : ±{args.tolerance_frames} frames")
    print(f"{'=' * 56}")

    print(f"\n--- Frame-level Detection ---")
    print(f"  Annotated frames : {dm.n_annotated_frames}")
    print(f"  Pipeline frames  : {dm.n_pipeline_frames}")
    print(f"  TP / FP / FN     : {dm.tp} / {dm.fp} / {dm.fn}")
    print(f"  Precision        : {dm.precision:.3f}")
    print(f"  Recall           : {dm.recall:.3f}")
    print(f"  F1               : {dm.f1:.3f}")
    print(f"  Recall (paired)  : {dm.recall_paired:.3f}")
    print(f"  Recall (single)  : {dm.recall_single:.3f}")

    print(f"\n--- Per-frame Count MAE (n={cm.n_frames_evaluated}) ---")
    print(f"  MAE total   : {cm.mae_total:.2f}")
    print(f"  MAE paired  : {cm.mae_paired:.2f}")
    print(f"  MAE single  : {cm.mae_single:.2f}")

    # ---- Spatial metrics (optional) ----
    if args.gt_positions:
        gt_positions: list[tuple[float, float]] = []
        for s in args.gt_positions:
            try:
                y_s, x_s = s.split(",")
                gt_positions.append((float(y_s), float(x_s)))
            except ValueError:
                logger.warning("Could not parse --gt-positions entry: %r", s)
        if gt_positions:
            sm = evaluate_spatial(pipeline_result, gt_positions, args.radius)
            print(f"\n--- Spatial Recall (radius={args.radius:.1f} px) ---")
            print(f"  GT positions  : {sm.n_gt}")
            print(f"  Matched       : {sm.n_matched}")
            print(f"  Recall        : {sm.recall:.3f}")
            if sm.unmatched_gt:
                print(f"  Unmatched GT  :")
                for gy, gx in sm.unmatched_gt:
                    print(f"    ({gy:.1f}, {gx:.1f})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
