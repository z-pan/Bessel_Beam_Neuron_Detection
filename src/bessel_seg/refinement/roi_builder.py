"""Build final NeuronROI objects from validated clusters and single-spot detections.

This module is the **dual-path merge point** of the pipeline.  It consumes:

1. **Paired clusters** — ``list[list[SpotPair]]`` from
   :func:`~bessel_seg.fusion.temporal_validate.validate_calcium_dynamics`.
   Each cluster represents one neuron seen as a pair across several frames.

2. **Single-spot neurons** — ``list[NeuronROI]`` from
   :func:`~bessel_seg.pairing.orphan_handler.detect_single_spot_neurons`.
   Each ROI was detected as a single fluorescent spot (the second Bessel
   lobe was below the noise floor in every frame).

For each detection the builder:

* Selects the top-K frames (by
  :class:`~bessel_seg.data_types.FrameQuality` score) in which the neuron
  was seen.
* Averages those frames of the raw stack to build a high-SNR reference image.
* Refines spot centres and widths by fitting a 2-D Gaussian to each spot
  in the reference image (via :func:`~bessel_seg.refinement.gaussian_fit.fit_spot_gaussian`).
* Constructs binary ROI masks (all pixels within 2 σ of the fitted centre).
* Extracts the ΔF/F₀ temporal trace from the combined mask region in *dff*.
* Computes the final confidence score (four-tier model).
* Assembles a :class:`~bessel_seg.data_types.NeuronROI` for each detection.

Neuron IDs are assigned sequentially: paired ROIs first (0…Np−1), then
single-spot ROIs (Np…Np+Ns−1).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from bessel_seg.config import RefinementConfig
from bessel_seg.data_types import FrameQuality, NeuronROI, Spot, SpotPair
from bessel_seg.refinement.confidence_score import compute_confidence
from bessel_seg.refinement.gaussian_fit import fit_spot_gaussian

logger = logging.getLogger(__name__)

# Mask radius multiplier: pixels within N * sigma of the centre are in the ROI
_MASK_SIGMA_MULT: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _circle_mask(H: int, W: int, cy: float, cx: float, radius: float) -> NDArray[np.bool_]:
    """Return a boolean (H, W) mask with True inside a circle.

    Args:
        H, W: Image dimensions.
        cy, cx: Centre (row, column).
        radius: Circle radius in pixels.

    Returns:
        (H, W) boolean array.
    """
    ys, xs = np.ogrid[:H, :W]
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    return dist2 <= radius ** 2  # type: ignore[return-value]


def _select_top_frames(
    candidate_frame_indices: list[int],
    frame_qualities: list[FrameQuality],
    top_k: int,
) -> list[int]:
    """Return the top-K frame indices sorted by quality score (descending).

    If *candidate_frame_indices* is non-empty, only frames in that list are
    considered (neuron-active frames).  If it is empty or fewer than 1 valid
    frame quality exists, all frames are used as candidates.

    Args:
        candidate_frame_indices: Frame indices where the neuron was detected.
        frame_qualities: All per-frame quality scores.
        top_k: Maximum number of frames to return.

    Returns:
        List of at most *top_k* frame indices, sorted by overall_score desc.
    """
    fq_by_idx = {fq.frame_idx: fq for fq in frame_qualities}

    if candidate_frame_indices:
        candidates = [
            fq_by_idx[fi] for fi in candidate_frame_indices if fi in fq_by_idx
        ]
    else:
        candidates = list(frame_qualities)

    if not candidates:
        # Last resort: first top_k frame indices available
        return sorted(candidate_frame_indices or [])[:top_k]

    candidates_sorted = sorted(candidates, key=lambda fq: fq.overall_score, reverse=True)
    return [fq.frame_idx for fq in candidates_sorted[:top_k]]


def _average_frames(
    stack: NDArray[np.float32],
    frame_indices: list[int],
) -> NDArray[np.float32]:
    """Return the mean image over the selected frames.

    Args:
        stack: (T, H, W) image stack.
        frame_indices: Non-empty list of frame indices.

    Returns:
        (H, W) float32 average image.
    """
    valid = [fi for fi in frame_indices if 0 <= fi < stack.shape[0]]
    if not valid:
        return stack.mean(axis=0)
    return stack[valid].mean(axis=0).astype(np.float32)


def _extract_temporal_trace(
    dff: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Extract mean ΔF/F₀ trace within a binary mask.

    Args:
        dff: (T, H, W) ΔF/F₀ stack.
        mask: (H, W) boolean ROI mask.

    Returns:
        (T,) float32 trace.  Returns zero trace if mask is all-False.
    """
    T = dff.shape[0]
    if not mask.any():
        return np.zeros(T, dtype=np.float32)
    # dff[:, mask] → (T, N_pixels); mean along pixel axis
    return dff[:, mask].mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_neuron_rois(
    clusters: list[list[SpotPair]],
    single_neurons: list[NeuronROI],
    stack: NDArray[np.float32],
    dff: NDArray[np.float32],
    frame_qualities: list[FrameQuality],
    config: RefinementConfig,
) -> list[NeuronROI]:
    """Build final NeuronROI objects from paired clusters and single-spot neurons.

    This is the **dual-path merge point**: both the paired detection path and
    the single-spot detection path feed into this function, which produces a
    unified list of :class:`~bessel_seg.data_types.NeuronROI` objects with
    fitted spot parameters, ROI masks, temporal traces, and confidence scores.

    Processing order:
        1. Paired clusters → NeuronROIs with ``detection_type="paired"``,
           IDs 0 … Np−1.
        2. Single-spot neurons → enriched NeuronROIs with IDs Np … Np+Ns−1.

    Args:
        clusters: Validated paired clusters from
            :func:`~bessel_seg.fusion.temporal_validate.validate_calcium_dynamics`.
            Each element is a list of :class:`~bessel_seg.data_types.SpotPair`
            objects from different frames.
        single_neurons: Single-spot
            :class:`~bessel_seg.data_types.NeuronROI` objects from
            :func:`~bessel_seg.pairing.orphan_handler.detect_single_spot_neurons`.
        stack: (T, H, W) registered + photobleach-corrected raw stack.
        dff: (T, H, W) ΔF/F₀ stack.
        frame_qualities: Per-frame quality scores (used to select top-K
            frames for averaging and Gaussian fitting).
        config: :class:`~bessel_seg.config.RefinementConfig`.

    Returns:
        List of :class:`~bessel_seg.data_types.NeuronROI` objects, ordered
        paired first then single, with monotonically increasing ``neuron_id``.
        ROIs with ``confidence < config.min_confidence`` are **excluded**.
    """
    T, H, W = dff.shape
    rois: list[NeuronROI] = []

    # ------------------------------------------------------------------
    # Path A: Paired clusters
    # ------------------------------------------------------------------
    for cluster in clusters:
        if not cluster:
            continue

        # --- Frame selection ---
        frame_indices = [
            p.left.frame_idx for p in cluster
            if p.left.frame_idx is not None
        ]
        top_frames = _select_top_frames(frame_indices, frame_qualities, config.top_k_frames)

        # --- Reference image ---
        avg_img = _average_frames(stack, top_frames)

        # --- Weighted mean centre (equal weight) ---
        centres = np.array([p.center for p in cluster], dtype=np.float64)  # (N, 2)
        cy, cx = centres.mean(axis=0)

        left_centres = np.array([(p.left.y, p.left.x) for p in cluster], dtype=np.float64)
        right_centres = np.array([(p.right.y, p.right.x) for p in cluster], dtype=np.float64)
        mean_left_y, mean_left_x = left_centres.mean(axis=0)
        mean_right_y, mean_right_x = right_centres.mean(axis=0)

        # --- Gaussian fit ---
        y0_l, x0_l, sigma_l, amp_l = fit_spot_gaussian(
            avg_img, (mean_left_y, mean_left_x), config.gaussian_fit_radius
        )
        y0_r, x0_r, sigma_r, amp_r = fit_spot_gaussian(
            avg_img, (mean_right_y, mean_right_x), config.gaussian_fit_radius
        )

        # --- Binary masks (2σ radius) ---
        mask_l = _circle_mask(H, W, y0_l, x0_l, _MASK_SIGMA_MULT * sigma_l)
        mask_r = _circle_mask(H, W, y0_r, x0_r, _MASK_SIGMA_MULT * sigma_r)
        combined_mask = mask_l | mask_r

        # --- Temporal trace ---
        trace = _extract_temporal_trace(dff, combined_mask)

        # --- Confidence ---
        confidence = compute_confidence(cluster, frame_qualities, trace, T)
        if confidence < config.min_confidence:
            logger.debug(
                "Paired cluster @ (%.1f, %.1f): confidence=%.3f < %.3f — skipped",
                cy, cx, confidence, config.min_confidence,
            )
            continue

        # --- Left / right Spot objects (refined) ---
        left_spot_ref = Spot(
            y=y0_l, x=x0_l,
            sigma=sigma_l,
            intensity=float(amp_l),
        )
        right_spot_ref = Spot(
            y=y0_r, x=x0_r,
            sigma=sigma_r,
            intensity=float(amp_r),
        )

        # Unique frame indices where this neuron was detected
        unique_frames = set(frame_indices) if frame_indices else set()

        roi = NeuronROI(
            neuron_id=len(rois),
            center_y=float(cy),
            center_x=float(cx),
            left_spot=left_spot_ref,
            right_spot=right_spot_ref,
            left_radius=float(_MASK_SIGMA_MULT * sigma_l),
            right_radius=float(_MASK_SIGMA_MULT * sigma_r),
            confidence=confidence,
            detection_type="paired",
            detection_count=len(unique_frames) if unique_frames else len(cluster),
            total_frames=T,
            temporal_trace=trace,
            mask_left=mask_l,
            mask_right=mask_r,
        )
        rois.append(roi)

    n_paired = len(rois)

    # ------------------------------------------------------------------
    # Path B: Single-spot neurons — enrich with fitted params & trace
    # ------------------------------------------------------------------
    for single in single_neurons:
        if single.left_spot is None:
            # Degenerate: no spot info at all, skip
            continue

        # --- Frame selection: single ROIs don't track frame indices easily;
        #     use global top-K frames for the average image. ---
        top_frames = _select_top_frames([], frame_qualities, config.top_k_frames)
        avg_img = _average_frames(stack, top_frames)

        # --- Gaussian fit on the single spot ---
        y0, x0, sigma_fit, amp_fit = fit_spot_gaussian(
            avg_img, (single.center_y, single.center_x), config.gaussian_fit_radius
        )

        # --- Binary mask (2σ radius) ---
        mask_s = _circle_mask(H, W, y0, x0, _MASK_SIGMA_MULT * sigma_fit)

        # --- Temporal trace ---
        trace = _extract_temporal_trace(dff, mask_s)

        # --- Confidence (recomputed with actual trace) ---
        confidence = compute_confidence(single, frame_qualities, trace, T)
        if confidence < config.min_confidence:
            logger.debug(
                "Single neuron @ (%.1f, %.1f) [%s]: confidence=%.3f < %.3f — skipped",
                single.center_y, single.center_x,
                single.detection_type, confidence, config.min_confidence,
            )
            continue

        # Refined left spot
        left_spot_ref = Spot(
            y=y0, x=x0,
            sigma=sigma_fit,
            intensity=float(amp_fit),
        )

        roi = NeuronROI(
            neuron_id=len(rois),
            center_y=y0,
            center_x=x0,
            left_spot=left_spot_ref,
            right_spot=None,
            left_radius=float(_MASK_SIGMA_MULT * sigma_fit),
            right_radius=0.0,
            confidence=confidence,
            detection_type=single.detection_type,
            detection_count=single.detection_count,
            total_frames=T,
            temporal_trace=trace,
            mask_left=mask_s,
            mask_right=None,
        )
        rois.append(roi)

    n_single = len(rois) - n_paired

    logger.info(
        "build_neuron_rois: %d paired clusters → %d paired ROIs; "
        "%d single-spot neurons → %d single ROIs; "
        "total %d (min_confidence=%.2f)",
        len(clusters), n_paired,
        len(single_neurons), n_single,
        len(rois), config.min_confidence,
    )
    return rois
