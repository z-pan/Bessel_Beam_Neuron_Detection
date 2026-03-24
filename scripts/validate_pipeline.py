#!/usr/bin/env python3
"""End-to-end pipeline validation on synthetic data.

Generates a synthetic Bessel beam dataset with known ground-truth neurons,
runs the full pipeline, and reports whether all expected neurons are detected
with correct properties.  This script requires NO external data files.

Usage
-----
.. code-block:: bash

    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --output /tmp/validation_results
    python scripts/validate_pipeline.py --verbose

Exit codes
----------
0 — all validation checks passed
1 — one or more checks failed
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Ground-truth specification
# ---------------------------------------------------------------------------
_SIGMA_SPOT: float = 2.5
_PAIR_DX: float = 43.0
_T, _H, _W = 60, 100, 160

# Ground-truth neuron locations (y, x) and type
GROUND_TRUTH = {
    "Neuron_A": {"center": (50.0, 80.0), "type": "paired"},
    "Neuron_B": {"center": (30.0, 110.0), "type": "paired"},
    "Neuron_C": {"center": (75.0, 40.0), "type": "single"},
    "Neuron_D": {"center": (20.0, 40.0), "type": "single"},
}
BG_BLOB = {"center": (60.0, 148.0), "type": "artifact"}
MATCH_RADIUS = 15.0  # px tolerance for position matching


def _gaussian_footprint(H: int, W: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    Y, X = np.ogrid[:H, :W]
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def _add_events(
    stack: np.ndarray, cy: float, cx: float, sigma: float,
    amplitude: float, starts: list[int], duration: int,
) -> None:
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


def make_validation_stack(seed: int = 42) -> np.ndarray:
    """Generate the synthetic validation stack.

    Contains:
    - 2 paired neurons (A, B) with both lobes visible
    - 2 single-spot neurons (C, D) with only one lobe
    - 1 persistent background blob (should be rejected)
    - Linear photobleaching (-25%)
    - Non-uniform background + horizontal band artifacts
    """
    np.random.seed(seed)

    bg = np.ones((_H, _W), dtype=np.float32) * 12.0
    bg += np.linspace(3.0, 0.0, _W, dtype=np.float32)[np.newaxis, :]
    for row in range(0, _H, 18):
        bg[row : row + 2, :] += 4.0

    bleach = np.linspace(1.0, 0.75, _T, dtype=np.float32)[:, np.newaxis, np.newaxis]
    stack = np.tile(bg[np.newaxis], (_T, 1, 1)) * bleach
    stack += np.random.normal(0.0, 0.3, (_T, _H, _W)).astype(np.float32)
    stack = np.clip(stack, 0.0, None).astype(np.float32)

    # Neuron A (paired)
    _add_events(stack, 50.0, 80.0 - _PAIR_DX / 2, _SIGMA_SPOT, 30.0, [5, 25, 45], 5)
    _add_events(stack, 50.0, 80.0 + _PAIR_DX / 2, _SIGMA_SPOT, 30.0, [5, 25, 45], 5)

    # Neuron B (paired)
    _add_events(stack, 30.0, 110.0 - _PAIR_DX / 2, _SIGMA_SPOT, 30.0, [10, 30, 50], 5)
    _add_events(stack, 30.0, 110.0 + _PAIR_DX / 2, _SIGMA_SPOT, 30.0, [10, 30, 50], 5)

    # Neuron C (single)
    _add_events(stack, 75.0, 40.0, _SIGMA_SPOT, 28.0, [15, 35, 53], 5)

    # Neuron D (single)
    _add_events(stack, 20.0, 40.0, _SIGMA_SPOT, 28.0, [8, 28, 48], 5)

    # Persistent background blob
    stack += 20.0 * _gaussian_footprint(_H, _W, 60.0, 148.0, 3.0)[np.newaxis, :, :]

    return stack.astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline validation on synthetic data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save pipeline outputs (temporary if omitted).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _nearest_neuron(neurons, cy: float, cx: float):
    """Find the nearest detected neuron to (cy, cx)."""
    if not neurons:
        return float("inf"), None
    dists = [math.hypot(r.center_y - cy, r.center_x - cx) for r in neurons]
    i = int(np.argmin(dists))
    return dists[i], neurons[i]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("validate_pipeline")

    from bessel_seg.config import PipelineConfig
    from bessel_seg.pipeline import run_pipeline

    # Configure pipeline for synthetic data
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

    # Generate synthetic data
    logger.info("Generating synthetic validation stack (%d, %d, %d)...", _T, _H, _W)
    stack = make_validation_stack(seed=42)

    # Write to temp file and run pipeline
    use_tmpdir = args.output is None
    if use_tmpdir:
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="bessel_validate_")
        work_dir = Path(tmpdir_obj.name)
    else:
        work_dir = Path(args.output)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmpdir_obj = None

    stack_path = work_dir / "validation_stack.tif"
    tifffile.imwrite(str(stack_path), stack)
    output_dir = work_dir / "results"

    logger.info("Running pipeline...")
    result = run_pipeline(str(stack_path), config=cfg, output_dir=str(output_dir))

    # ---------------------------------------------------------------
    # Validation checks
    # ---------------------------------------------------------------
    checks_passed = 0
    checks_failed = 0
    checks_total = 0

    def check(description: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            logger.info("  PASS: %s", description)
        else:
            checks_failed += 1
            msg = f"  FAIL: {description}"
            if detail:
                msg += f" — {detail}"
            logger.error(msg)

    print(f"\n{'=' * 60}")
    print("  PIPELINE VALIDATION RESULTS")
    print(f"{'=' * 60}\n")

    # --- Check 1: Result structure ---
    from bessel_seg.data_types import PipelineResult
    check("Returns PipelineResult", isinstance(result, PipelineResult))
    check("Has neurons list", isinstance(result.neurons, list))
    check("Has frame_qualities", isinstance(result.frame_qualities, list))
    check("Background mask shape", result.background_mask.shape == (_H, _W))
    check("Summary maps present",
          all(k in result.summary_maps for k in ("sigma", "max_proj", "mean", "cv", "edi")))

    # --- Check 2: Neuron count ---
    n = len(result.neurons)
    check(f"Detected {n} neurons (expected 4)", n == 4,
          f"got {n}")

    # --- Check 3: Type distribution ---
    n_paired = sum(1 for r in result.neurons if r.detection_type == "paired")
    n_single = n - n_paired
    check(f"2 paired + 2 single (got {n_paired}p + {n_single}s)",
          n_paired == 2 and n_single == 2)

    # --- Check 4: Location matching ---
    for name, gt in GROUND_TRUTH.items():
        cy, cx = gt["center"]
        d, roi = _nearest_neuron(result.neurons, cy, cx)
        check(f"{name} detected within {MATCH_RADIUS} px",
              d < MATCH_RADIUS,
              f"nearest distance: {d:.1f} px")

    # --- Check 5: Background blob rejected ---
    bg_cy, bg_cx = BG_BLOB["center"]
    d, _ = _nearest_neuron(result.neurons, bg_cy, bg_cx)
    check("Background blob rejected",
          d > 10.0,
          f"nearest distance: {d:.1f} px")

    # --- Check 6: Confidence tiers ---
    paired = [r for r in result.neurons if r.detection_type == "paired"]
    single = [r for r in result.neurons if r.detection_type != "paired"]
    if paired:
        min_paired_conf = min(r.confidence for r in paired)
        check(f"Paired confidence >= 0.70 (min={min_paired_conf:.3f})",
              min_paired_conf >= 0.70)
    if paired and single:
        mean_p = float(np.mean([r.confidence for r in paired]))
        mean_s = float(np.mean([r.confidence for r in single]))
        check(f"Mean paired conf ({mean_p:.3f}) > mean single conf ({mean_s:.3f})",
              mean_p > mean_s)

    # --- Check 7: ROI properties ---
    for roi in result.neurons:
        check(f"Neuron {roi.neuron_id} confidence in [0, 1]",
              0.0 <= roi.confidence <= 1.0)
        check(f"Neuron {roi.neuron_id} centre in image bounds",
              0 <= roi.center_y < _H and 0 <= roi.center_x < _W)
        if roi.temporal_trace is not None:
            check(f"Neuron {roi.neuron_id} trace shape ({_T},)",
                  roi.temporal_trace.shape == (_T,))

    # --- Check 8: Output files ---
    for fname in ("neurons.json", "roi_masks.tif", "traces.csv",
                   "frame_quality.csv", "summary_maps.npz", "report.html"):
        check(f"Output file {fname} exists", (output_dir / fname).exists())

    # --- Summary ---
    print(f"\n{'=' * 60}")
    status = "ALL PASSED" if checks_failed == 0 else "SOME FAILED"
    print(f"  {status}: {checks_passed}/{checks_total} checks passed")
    if checks_failed > 0:
        print(f"  {checks_failed} checks FAILED")
    print(f"{'=' * 60}\n")

    if not use_tmpdir:
        print(f"  Results saved to: {output_dir}\n")

    if tmpdir_obj is not None:
        tmpdir_obj.cleanup()

    return 0 if checks_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
