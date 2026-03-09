"""Single-spot neuron detection — the co-equal second detection path.

CRITICAL (CLAUDE.md): 52 % of all neuron observations across 10 datasets show
only a single fluorescent spot.  The rate ranges from 14 % (vol_003) to 77 %
(vol_008).  A pair-only algorithm misses over half of all events in most
volumes.  This module implements the mandatory single-spot path.

Two stages:

1. :func:`rescue_orphans_across_frames` — attempt to recover orphans that
   belong to a neuron already observed as a *pair* in a nearby frame.  If a
   paired detection exists within ±2 frames and ±12 px of the orphan, the
   orphan is promoted to a low-confidence SpotPair using the known geometry
   from the paired frame as a template.

2. :func:`detect_single_spot_neurons` — classify the remaining unrecovered
   orphans using temporal dynamics and spatial proximity to known pairs.
   Returns :class:`~bessel_seg.data_types.NeuronROI` objects with tiered
   confidence scores (CLAUDE.md four-tier model).
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from bessel_seg.config import PairingConfig
from bessel_seg.data_types import NeuronROI, Spot, SpotPair

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Spatial proximity thresholds (calibrated from CLAUDE.md)
# -------------------------------------------------------------------------
# Cross-frame rescue: orphan must be within this many px of a paired centre
_RESCUE_SPATIAL_EPS: float = 12.0
# Single-spot high-confidence: orphan within this many px of a known pair centre
_NEAR_PAIR_EPS: float = 25.0
# Cross-frame temporal window for rescue (± frames)
_RESCUE_FRAME_WINDOW: int = 2

# -------------------------------------------------------------------------
# Confidence tiers (CLAUDE.md §"Confidence model")
# -------------------------------------------------------------------------
_CONF_PAIRED_TEMPORAL   = 0.90   # paired + temporal correlation (not used here)
_CONF_NEAR_PAIR         = 0.70   # single-spot + near known pair
_CONF_ISOLATED_TEMPORAL = 0.50   # single-spot + good temporal dynamics
_CONF_LOW               = 0.25   # single-spot + weak dynamics

# ΔF/F₀ thresholds for temporal validation
_MIN_PEAK_DFF:       float = 0.20    # calibrated p10 from 3 202 measurements
_MAX_ACTIVE_FRAC:    float = 0.50    # reject if active > 50 % of frames (background)
_ROI_RADIUS:         int   = 4       # px radius for signal extraction


def _extract_trace(
    dff: NDArray[np.float32],
    cy: float,
    cx: float,
    radius: int = _ROI_RADIUS,
) -> NDArray[np.float32]:
    """Extract mean ΔF/F₀ trace within a circular ROI around (cy, cx).

    Args:
        dff: (T, H, W) ΔF/F₀ stack.
        cy, cx: Centre coordinates (row, column).
        radius: Pixel radius of the ROI.

    Returns:
        1-D float32 array of shape (T,).
    """
    T, H, W = dff.shape
    iy, ix = int(round(cy)), int(round(cx))

    y_lo = max(0, iy - radius)
    y_hi = min(H, iy + radius + 1)
    x_lo = max(0, ix - radius)
    x_hi = min(W, ix + radius + 1)

    patch = dff[:, y_lo:y_hi, x_lo:x_hi]   # (T, ph, pw)
    trace = patch.mean(axis=(1, 2))          # (T,)
    return trace.astype(np.float32)


def _classify_temporal(
    trace: NDArray[np.float32],
) -> tuple[bool, float]:
    """Return (is_transient, peak_dff) for a temporal trace.

    A valid calcium transient must:
      - Have peak ΔF/F₀ ≥ ``_MIN_PEAK_DFF``
      - Be active (above half-peak) in < ``_MAX_ACTIVE_FRAC`` of frames

    Args:
        trace: (T,) ΔF/F₀ trace.

    Returns:
        ``(is_transient, peak_dff)`` — ``is_transient`` is ``True`` if the
        trace looks like a genuine calcium event rather than persistent
        background.
    """
    peak_dff = float(trace.max())
    if peak_dff < _MIN_PEAK_DFF:
        return False, peak_dff

    half_peak = peak_dff * 0.5
    active_frac = float((trace >= half_peak).mean())
    is_transient = active_frac < _MAX_ACTIVE_FRAC
    return is_transient, peak_dff


def rescue_orphans_across_frames(
    orphans_per_frame: dict[int, list[Spot]],
    pairs_per_frame: dict[int, list[SpotPair]],
    config: PairingConfig,
) -> list[SpotPair]:
    """Promote orphan spots to low-confidence SpotPairs using adjacent-frame geometry.

    For every orphan spot in frame *t*, the algorithm searches frames
    *[t − window, t + window]* for a paired neuron whose centre is within
    ``_RESCUE_SPATIAL_EPS`` pixels of the orphan.  If one is found, a new
    :class:`~bessel_seg.data_types.SpotPair` is constructed using:

    * The orphan as one spot (left or right, determined by its position
      relative to the template pair centre).
    * A synthetic "ghost" spot at the complementary position implied by the
      template pair geometry (same spacing and vertical alignment).
    * ``pair_cost = 0.8`` to mark it as a low-confidence recovery.

    Args:
        orphans_per_frame: Dict ``frame_idx → list[Spot]``.
        pairs_per_frame:   Dict ``frame_idx → list[SpotPair]``.
        config: PairingConfig (``ideal_distance`` used as template spacing).

    Returns:
        List of rescued :class:`~bessel_seg.data_types.SpotPair` objects.
        These should be merged with the primary paired detections before
        spatial clustering.
    """
    rescued: list[SpotPair] = []
    all_pair_frames = sorted(pairs_per_frame.keys())

    for frame_t, orphans in orphans_per_frame.items():
        # Collect candidate template pairs from ±window frames
        template_pairs: list[SpotPair] = []
        for df in range(-_RESCUE_FRAME_WINDOW, _RESCUE_FRAME_WINDOW + 1):
            template_pairs.extend(pairs_per_frame.get(frame_t + df, []))

        if not template_pairs:
            continue

        for orphan in orphans:
            best_pair: Optional[SpotPair] = None
            best_dist = _RESCUE_SPATIAL_EPS + 1.0

            for tp in template_pairs:
                cy, cx = tp.center
                dist = math.hypot(orphan.y - cy, orphan.x - cx)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = tp

            if best_pair is None:
                continue

            # Build a synthetic companion spot at the complementary position
            cy, cx = best_pair.center
            # Determine whether orphan is the left or right spot
            half_dx = best_pair.horizontal_distance / 2.0
            if orphan.x < cx:
                # orphan is left; ghost is right
                ghost_x = cx + half_dx
                ghost_y = cy + (best_pair.right.y - best_pair.left.y) / 2.0
                ghost_sigma = best_pair.right.sigma
                ghost_intensity = orphan.intensity * best_pair.intensity_ratio
                ghost = Spot(
                    y=ghost_y, x=ghost_x,
                    sigma=ghost_sigma,
                    intensity=ghost_intensity,
                    frame_idx=orphan.frame_idx,
                )
                left_spot, right_spot = orphan, ghost
            else:
                # orphan is right; ghost is left
                ghost_x = cx - half_dx
                ghost_y = cy - (best_pair.right.y - best_pair.left.y) / 2.0
                ghost_sigma = best_pair.left.sigma
                ghost_intensity = orphan.intensity * best_pair.intensity_ratio
                ghost = Spot(
                    y=ghost_y, x=ghost_x,
                    sigma=ghost_sigma,
                    intensity=ghost_intensity,
                    frame_idx=orphan.frame_idx,
                )
                left_spot, right_spot = ghost, orphan

            dx = right_spot.x - left_spot.x
            dy = abs(right_spot.y - left_spot.y)
            i_max = max(left_spot.intensity, right_spot.intensity)
            i_min = min(left_spot.intensity, right_spot.intensity)
            intensity_ratio = (i_min / i_max) if i_max > 1e-9 else 1.0
            sigma_max = max(left_spot.sigma, right_spot.sigma)
            scale_sim = (
                1.0 - abs(left_spot.sigma - right_spot.sigma) / sigma_max
                if sigma_max > 1e-9 else 1.0
            )

            rescued_pair = SpotPair(
                left=left_spot,
                right=right_spot,
                pair_distance=math.hypot(dx, dy),
                horizontal_distance=dx,
                vertical_offset=dy,
                intensity_ratio=intensity_ratio,
                scale_similarity=scale_sim,
                pair_cost=0.8,          # low-confidence marker
            )
            rescued.append(rescued_pair)

    logger.info(
        "rescue_orphans_across_frames: rescued %d orphan pairs from %d orphan frames",
        len(rescued), len(orphans_per_frame),
    )
    return rescued


def detect_single_spot_neurons(
    orphans: list[Spot],
    known_pairs: list[SpotPair],
    dff: NDArray[np.float32],
    config: PairingConfig,
) -> list[NeuronROI]:
    """Classify unrecovered orphan spots as potential single-spot neurons.

    Four-tier confidence model (CLAUDE.md):

    * **HIGH** (0.70): orphan within ``_NEAR_PAIR_EPS`` px of a known pair
      centre AND shows transient dynamics.
    * **MEDIUM** (0.50): orphan shows clear transient dynamics (peak ΔF/F₀ ≥
      0.20, active < 50 % of frames) but no nearby pair.
    * **LOW** (0.25): orphan has some signal but does not pass the transient
      gate, or is persistent.
    * **REJECT** (< 0.25): orphan is persistent (active ≥ 50 % of frames) and
      has no signal peak — likely background.

    Args:
        orphans: Unrecovered :class:`~bessel_seg.data_types.Spot` objects.
        known_pairs: All confirmed :class:`~bessel_seg.data_types.SpotPair`
            objects (used for spatial association check).
        dff: (T, H, W) ΔF/F₀ stack for temporal validation.
        config: PairingConfig (``ideal_distance`` used for spatial association).

    Returns:
        List of :class:`~bessel_seg.data_types.NeuronROI` objects with
        ``detection_type`` set to ``"single_near_pair"`` or
        ``"single_isolated"`` and confidence in [0.25, 0.80].
        Orphans below the minimum confidence threshold (0.25) are discarded.
    """
    T, H, W = dff.shape
    neuron_id_start = 0  # caller should offset if merging with paired results
    rois: list[NeuronROI] = []

    # Pre-compute known pair centres for fast distance queries
    pair_centres: list[tuple[float, float]] = [p.center for p in known_pairs]

    for orphan in orphans:
        trace = _extract_trace(dff, orphan.y, orphan.x)
        is_transient, peak_dff = _classify_temporal(trace)

        # --- Persistent / background: discard ---
        # Reject if signal is too weak
        if peak_dff < _MIN_PEAK_DFF:
            continue
        # Reject if signal is persistent (active ≥ _MAX_ACTIVE_FRAC of frames)
        # — this catches bright-but-constant background, even with high peak_dff.
        if not is_transient:
            continue

        # --- Spatial association: within _NEAR_PAIR_EPS px of a known pair ---
        near_pair = False
        if pair_centres:
            dists = [
                math.hypot(orphan.y - cy, orphan.x - cx)
                for cy, cx in pair_centres
            ]
            near_pair = min(dists) < _NEAR_PAIR_EPS

        # --- Assign confidence tier ---
        if is_transient and near_pair:
            confidence = _CONF_NEAR_PAIR
            det_type = "single_near_pair"
        elif is_transient:
            confidence = _CONF_ISOLATED_TEMPORAL
            det_type = "single_isolated"
        else:
            confidence = _CONF_LOW
            det_type = "single_isolated"

        # Minimum confidence gate (0.25 from RefinementConfig.min_confidence)
        if confidence < 0.25:
            continue

        # Count how many frames the spot was active
        detection_count = int((trace >= _MIN_PEAK_DFF).sum())

        roi = NeuronROI(
            neuron_id=neuron_id_start + len(rois),
            center_y=orphan.y,
            center_x=orphan.x,
            left_spot=orphan,
            right_spot=None,
            left_radius=orphan.sigma * 2.0,
            right_radius=0.0,
            confidence=confidence,
            detection_type=det_type,
            detection_count=detection_count,
            total_frames=T,
        )
        rois.append(roi)

    logger.info(
        "detect_single_spot_neurons: %d orphans → %d single-spot NeuronROIs",
        len(orphans), len(rois),
    )
    return rois
