"""Main pipeline orchestration for Bessel beam neuron segmentation.

This module ties together all six processing modules and exposes a single
:func:`run_pipeline` entry point.  Processing order follows CLAUDE.md:

    1.  Load data (folder-of-frames **or** multi-frame .tif)
    2.  Module 1 — Preprocessing:
          register → photobleach-correct → denoise
    3.  Module 2 — Temporal analysis:
          baseline → ΔF/F₀ → summary maps → illumination mask
    4.  Module 3 — Detection:
          build EDI → per-frame spot detect → adaptive filter
    5.  Module 4 — Pairing (PRIMARY path):
          per-frame geometric pairing → Hungarian solver
    6.  Module 4b — Orphan handling (SECONDARY path):
          cross-frame rescue → single-spot neuron classification
    7.  Module 5 — Fusion:
          frame quality scoring → DBSCAN spatial clustering → temporal validation
    8.  Module 6 — Refinement:
          Gaussian fit → build ROIs with traces → confidence scores
    9.  Save outputs
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from numpy.typing import NDArray
from tqdm import tqdm

from bessel_seg.config import PipelineConfig
from bessel_seg.data_types import NeuronROI, PipelineResult, Spot, SpotPair

# Module 1
from bessel_seg.preprocessing.channel_extract import extract_green_channel
from bessel_seg.preprocessing.photobleach import correct_photobleaching
from bessel_seg.preprocessing.registration import rigid_register
from bessel_seg.preprocessing.denoise import deepcad_denoise

# Module 2
from bessel_seg.temporal.background_mask import generate_illumination_mask
from bessel_seg.temporal.baseline import estimate_baseline
from bessel_seg.temporal.delta_f import compute_delta_f
from bessel_seg.temporal.summary_maps import compute_summary_maps

# Module 3
from bessel_seg.detection.adaptive_threshold import filter_spots_adaptive
from bessel_seg.detection.blob_detect import detect_spots_per_frame
from bessel_seg.detection.enhanced_image import build_enhanced_detection_image

# Module 4
from bessel_seg.pairing.geometric_match import find_candidate_pairs
from bessel_seg.pairing.hungarian_solver import solve_optimal_pairing
from bessel_seg.pairing.orphan_handler import (
    detect_single_spot_neurons,
    rescue_orphans_across_frames,
)

# Module 5
from bessel_seg.fusion.frame_quality import compute_frame_quality
from bessel_seg.fusion.spatial_cluster import cluster_neuron_detections
from bessel_seg.fusion.temporal_validate import validate_calcium_dynamics

# Module 6
from bessel_seg.refinement.roi_builder import build_neuron_rois

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_data(data_path: Path) -> NDArray[np.float32]:
    """Load a dataset as a (T, H, W) float32 stack.

    Supports two layouts:
    * Directory of single-frame .tif files — frames sorted by filename.
    * Multi-frame .tif file — loaded via :func:`extract_green_channel`.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if data_path.is_dir():
        tif_files = sorted(
            list(data_path.glob("*.tif")) + list(data_path.glob("*.tiff"))
        )
        if not tif_files:
            raise ValueError(f"No .tif/.tiff files found in: {data_path}")
        logger.info("Loading %d frames from directory %s", len(tif_files), data_path)
        frames = [tifffile.imread(str(f)).astype(np.float32) for f in tif_files]
        stack = np.stack(frames, axis=0)
        if stack.ndim == 4:          # (T, H, W, C) RGB → green channel
            logger.info("Extracting green channel from (T,H,W,C) stack")
            stack = stack[:, :, :, 1]
        return stack.astype(np.float32)

    if data_path.suffix.lower() in {".tif", ".tiff"}:
        return extract_green_channel(data_path)

    raise ValueError(
        f"Unsupported path: expected .tif file or directory, got {data_path}"
    )


def _deduplicate_orphans(orphans: list[Spot], eps: float) -> list[Spot]:
    """Collapse spatially-nearby orphan spots to one representative per neighbourhood.

    Multiple active frames produce orphan detections at the same physical
    location (same neuron, different frames).  Before single-spot classification
    we keep at most one spot per eps-radius cluster, choosing the spot with the
    highest SNR (or intensity as fallback).

    Args:
        orphans: All orphan spots from all frames (may repeat locations).
        eps: Deduplication radius (pixels).

    Returns:
        Deduplicated list with at most one spot per spatial neighbourhood.
    """
    if not orphans:
        return []

    def _quality(s: Spot) -> float:
        return s.snr if s.snr > 0 else s.intensity

    sorted_spots = sorted(orphans, key=_quality, reverse=True)
    positions = np.array([(s.y, s.x) for s in sorted_spots], dtype=np.float64)
    eps2 = eps ** 2

    kept: list[Spot] = []
    used = np.zeros(len(sorted_spots), dtype=bool)

    for i, spot in enumerate(sorted_spots):
        if used[i]:
            continue
        kept.append(spot)
        dists2 = (positions[:, 0] - spot.y) ** 2 + (positions[:, 1] - spot.x) ** 2
        used |= dists2 < eps2

    return kept


def _spot_to_dict(spot: Optional[Spot]) -> Optional[dict]:
    if spot is None:
        return None
    return {
        "y": float(spot.y),
        "x": float(spot.x),
        "sigma": float(spot.sigma),
        "intensity": float(spot.intensity),
        "snr": float(spot.snr),
    }


def _roi_to_dict(roi: NeuronROI) -> dict:
    return {
        "neuron_id": roi.neuron_id,
        "center_y": float(roi.center_y),
        "center_x": float(roi.center_x),
        "left_spot": _spot_to_dict(roi.left_spot),
        "right_spot": _spot_to_dict(roi.right_spot),
        "left_radius": float(roi.left_radius),
        "right_radius": float(roi.right_radius),
        "confidence": float(roi.confidence),
        "detection_type": roi.detection_type,
        "detection_count": roi.detection_count,
        "total_frames": roi.total_frames,
    }


def _generate_html_report(result: PipelineResult, output_dir: Path) -> None:
    n_total = len(result.neurons)
    n_paired = sum(1 for r in result.neurons if r.detection_type == "paired")
    n_single = n_total - n_paired
    mean_conf = float(np.mean([r.confidence for r in result.neurons])) if result.neurons else 0.0
    T = result.neurons[0].total_frames if result.neurons else 0
    mean_fq = (
        float(np.mean([fq.overall_score for fq in result.frame_qualities]))
        if result.frame_qualities else 0.0
    )

    rows = "\n".join(
        f"<tr><td>{r.neuron_id}</td><td>{r.center_y:.1f}</td><td>{r.center_x:.1f}</td>"
        f"<td>{r.detection_type}</td><td>{r.confidence:.3f}</td><td>{r.detection_count}</td></tr>"
        for r in result.neurons
    )
    meta_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in result.metadata.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Bessel Beam Segmentation Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:2em}}
h1{{color:#2c5f8a}}h2{{color:#444;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;width:100%;margin-bottom:1.5em}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#e8f0f8}}
.stat{{display:inline-block;background:#f4f8ff;border:1px solid #c0d4f0;
       border-radius:6px;padding:10px 20px;margin:6px;text-align:center}}
.val{{font-size:2em;font-weight:bold;color:#2c5f8a}}
.lbl{{font-size:.85em;color:#666}}
</style></head><body>
<h1>Bessel Beam Neuron Segmentation — Quality Report</h1>
<h2>Summary</h2>
<div>
<div class="stat"><div class="val">{n_total}</div><div class="lbl">Total neurons</div></div>
<div class="stat"><div class="val">{n_paired}</div><div class="lbl">Paired</div></div>
<div class="stat"><div class="val">{n_single}</div><div class="lbl">Single-spot</div></div>
<div class="stat"><div class="val">{mean_conf:.2f}</div><div class="lbl">Mean confidence</div></div>
<div class="stat"><div class="val">{T}</div><div class="lbl">Frames</div></div>
<div class="stat"><div class="val">{mean_fq:.2f}</div><div class="lbl">Mean frame quality</div></div>
</div>
<h2>Detected Neurons</h2>
<table><tr><th>ID</th><th>Y</th><th>X</th><th>Type</th><th>Conf</th><th>Detections</th></tr>
{rows}</table>
<h2>Processing Metadata</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{meta_rows}</table>
</body></html>"""

    (output_dir / "report.html").write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s/report.html", output_dir)


def _save_results(result: PipelineResult, output_dir: Path) -> None:
    """Save pipeline outputs: neurons.json, roi_masks.tif, traces.csv,
    frame_quality.csv, summary_maps.npz, report.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", output_dir)

    # neurons.json
    (output_dir / "neurons.json").write_text(
        json.dumps([_roi_to_dict(r) for r in result.neurons], indent=2),
        encoding="utf-8",
    )

    # roi_masks.tif  — (N, H, W) uint8, pixel value = neuron_id+1
    H, W = result.background_mask.shape
    if result.neurons:
        masks = np.zeros((len(result.neurons), H, W), dtype=np.uint8)
        for idx, roi in enumerate(result.neurons):
            if roi.mask_left is not None:
                masks[idx][roi.mask_left] = idx + 1
            if roi.mask_right is not None:
                masks[idx][roi.mask_right] = idx + 1
    else:
        masks = np.zeros((1, H, W), dtype=np.uint8)
    tifffile.imwrite(str(output_dir / "roi_masks.tif"), masks)

    # traces.csv
    T = result.neurons[0].total_frames if result.neurons else 0
    df = pd.DataFrame({"frame_idx": np.arange(T)})
    for roi in result.neurons:
        col = f"neuron_{roi.neuron_id}"
        if roi.temporal_trace is not None and roi.temporal_trace.shape[0] == T:
            df[col] = roi.temporal_trace.astype(float)
        else:
            df[col] = 0.0
    df.to_csv(output_dir / "traces.csv", index=False)

    # frame_quality.csv
    pd.DataFrame([
        {"frame_idx": fq.frame_idx, "snr": fq.snr, "sharpness": fq.sharpness,
         "pair_rate": fq.pair_rate, "overall_score": fq.overall_score}
        for fq in result.frame_qualities
    ]).to_csv(output_dir / "frame_quality.csv", index=False)

    # summary_maps.npz
    save_dict = {k: v for k, v in result.summary_maps.items()}
    save_dict["background_mask"] = result.background_mask.astype(np.uint8)
    np.savez(str(output_dir / "summary_maps.npz"), **save_dict)

    # report.html
    _generate_html_report(result, output_dir)
    logger.info("Results saved: %d neurons", len(result.neurons))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str | Path,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[str | Path] = None,
) -> PipelineResult:
    """Process one dataset through the full Bessel beam segmentation pipeline.

    Args:
        data_path: Directory of .tif frames **or** a multi-frame .tif/.tiff file.
        config: Typed pipeline configuration; uses defaults if ``None``.
        output_dir: Directory for saving results; if ``None``, results are
            returned but not written to disk.

    Returns:
        :class:`~bessel_seg.data_types.PipelineResult` with all neurons,
        frame qualities, summary maps, and processing metadata.
    """
    t_pipeline = time.time()

    # --- Config ---
    if config is None:
        config = PipelineConfig()
    data_path = Path(data_path)
    if output_dir is not None:
        output_dir = Path(output_dir)

    # ------------------------------------------------------------------ #
    # Step 2: Load data                                                    #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    stack = _load_data(data_path)
    T, H, W = stack.shape
    logger.info(
        "Data: T=%d, H=%d, W=%d, range=[%.1f, %.1f]  (%.1fs)",
        T, H, W, float(stack.min()), float(stack.max()), time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Module 1 — Preprocessing                                    #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    stack, _shifts = rigid_register(
        stack,
        method=config.preprocessing.registration_method,
        upsample_factor=config.preprocessing.registration_upsample,
    )
    stack = correct_photobleaching(stack, method=config.preprocessing.photobleach_correction)
    stack = deepcad_denoise(stack, config.deepcad)

    logger.info("Module 1 (preprocessing): %.1fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # Step 4: Module 2 — Temporal analysis                                #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    baseline = estimate_baseline(stack, percentile=config.preprocessing.baseline_percentile)
    dff = compute_delta_f(stack, baseline)
    summary_maps = compute_summary_maps(stack, dff)
    illumination_mask = generate_illumination_mask(
        stack, threshold_percentile=config.background.illumination_threshold_percentile
    )

    logger.info(
        "Module 2 (temporal): illumination mask %d/%d px valid  (%.1fs)",
        int(illumination_mask.sum()), H * W, time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Module 3 — Detection                                        #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    edi = build_enhanced_detection_image(summary_maps, illumination_mask, config.edi)

    # Per-frame detection on dff stack
    spots_per_frame_raw = detect_spots_per_frame(dff, illumination_mask, config.detection)

    # Adaptive local-SNR filtering per frame
    spots_per_frame: dict[int, list[Spot]] = {}
    for fi, spots in spots_per_frame_raw.items():
        spots_per_frame[fi] = filter_spots_adaptive(spots, dff[fi], config.detection) if spots else []

    total_spots = sum(len(s) for s in spots_per_frame.values())
    active_frames = sum(1 for s in spots_per_frame.values() if s)
    logger.info(
        "Module 3 (detection): %d spots in %d/%d active frames  (%.1fs)",
        total_spots, active_frames, T, time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 6: Module 4 — Per-frame geometric pairing                      #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    pairs_per_frame: dict[int, list[SpotPair]] = {}
    orphans_per_frame: dict[int, list[Spot]] = {}

    for fi, spots in tqdm(spots_per_frame.items(), desc="Pairing", leave=False):
        if not spots:
            pairs_per_frame[fi] = []
            orphans_per_frame[fi] = []
            continue
        candidates = find_candidate_pairs(spots, config.pairing)
        pairs, orphans = solve_optimal_pairing(spots, candidates)
        pairs_per_frame[fi] = pairs
        orphans_per_frame[fi] = orphans

    all_pairs: list[SpotPair] = [sp for pairs in pairs_per_frame.values() for sp in pairs]
    n_raw_orphans = sum(len(o) for o in orphans_per_frame.values())

    if total_spots > 0:
        pair_rate = 2 * len(all_pairs) / total_spots
        if pair_rate < 0.1:
            logger.warning("Low pair rate %.1f%% — many single-spot neurons likely", 100 * pair_rate)

    logger.info(
        "Module 4 primary: %d pairs, %d orphan spots  (%.1fs)",
        len(all_pairs), n_raw_orphans, time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 7: Module 4b — Orphan rescue + single-spot detection           #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    rescued = rescue_orphans_across_frames(orphans_per_frame, pairs_per_frame, config.pairing)
    for sp in rescued:
        fi = sp.left.frame_idx
        if fi is not None:
            pairs_per_frame.setdefault(fi, []).append(sp)
    all_pairs = [sp for pairs in pairs_per_frame.values() for sp in pairs]

    raw_orphans = [s for spots in orphans_per_frame.values() for s in spots]
    unique_orphans = _deduplicate_orphans(raw_orphans, eps=config.clustering.eps)
    single_neurons = detect_single_spot_neurons(unique_orphans, all_pairs, dff, config.pairing)

    logger.info(
        "Module 4b (orphan/single): %d rescued, %d unique orphans → %d single-spot ROIs  (%.1fs)",
        len(rescued), len(unique_orphans), len(single_neurons), time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 8: Module 5 — Fusion                                           #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    frame_qualities = compute_frame_quality(stack, spots_per_frame, pairs_per_frame)
    clusters = cluster_neuron_detections(all_pairs, config.clustering)
    validated_clusters = validate_calcium_dynamics(clusters, dff, config.temporal_validation)

    logger.info(
        "Module 5 (fusion): %d/%d clusters pass validation  (%.1fs)",
        len(validated_clusters), len(clusters), time.time() - t0,
    )

    # ------------------------------------------------------------------ #
    # Step 9: Module 6 — Refinement                                       #
    # ------------------------------------------------------------------ #
    t0 = time.time()

    neurons = build_neuron_rois(
        validated_clusters, single_neurons, stack, dff, frame_qualities, config.refinement
    )

    if not neurons:
        logger.warning("No neurons found — check detection/pairing/clustering thresholds")

    n_paired_roi = sum(1 for r in neurons if r.detection_type == "paired")
    n_single_roi = len(neurons) - n_paired_roi
    logger.info(
        "Module 6 (refinement): %d neurons (%d paired, %d single)  (%.1fs)",
        len(neurons), n_paired_roi, n_single_roi, time.time() - t0,
    )

    # --- Assemble result ---
    metadata: dict = {
        "data_path": str(data_path),
        "n_frames": T,
        "frame_shape": f"{H}x{W}",
        "registration_method": config.preprocessing.registration_method,
        "photobleach_correction": config.preprocessing.photobleach_correction,
        "n_spots_total": total_spots,
        "n_active_frames": active_frames,
        "n_pairs_total": len(all_pairs),
        "n_rescued_pairs": len(rescued),
        "n_clusters_raw": len(clusters),
        "n_clusters_validated": len(validated_clusters),
        "n_neurons_total": len(neurons),
        "n_neurons_paired": n_paired_roi,
        "n_neurons_single": n_single_roi,
        "pipeline_elapsed_s": round(time.time() - t_pipeline, 2),
    }

    result = PipelineResult(
        neurons=neurons,
        frame_qualities=frame_qualities,
        background_mask=~illumination_mask,
        summary_maps={**summary_maps, "edi": edi},
        metadata=metadata,
    )

    if output_dir is not None:
        _save_results(result, output_dir)

    logger.info(
        "Pipeline complete: %d neurons in %.1fs",
        len(neurons), time.time() - t_pipeline,
    )
    return result


def run_pipeline_dual_path(
    data_path: str | Path,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Enhanced pipeline merging EDI-based and per-frame detection paths.

    Currently delegates to :func:`run_pipeline` (which already uses per-frame
    detection as the primary path).  A future implementation will add a true
    EDI-based secondary detection path and merge the two result sets.
    """
    logger.info("run_pipeline_dual_path: delegating to run_pipeline")
    return run_pipeline(data_path, config=config)
