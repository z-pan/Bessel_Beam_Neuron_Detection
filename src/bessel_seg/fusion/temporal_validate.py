"""Temporal validation of candidate neuron clusters.

After spatial clustering, each cluster represents a candidate neuron.  This
module validates each cluster by examining whether the ΔF/F₀ trace at the
cluster centre exhibits genuine calcium-transient dynamics:

  * At least one peak above ``min_peak_dff`` (default 0.20).
  * At least ``min_rise_frames`` (default 1) of increasing signal before the
    peak — calcium transients have a characteristic rise.
  * At least ``min_decay_frames`` (default 1) of decreasing signal after the
    peak — calcium transients decay over several frames.
  * Active fraction below ``max_active_fraction`` (default 0.50) — persistent
    bright regions are background, not neurons.

Events are detected using a sliding window of ``event_window`` frames.
Individual events longer than ``max_event_duration`` frames are also rejected
as artefacts.

Real-data calibration (CLAUDE.md):
  - Events last 3–7 frames per annotation.
  - ΔF/F₀ p10 = 0.20 at confirmed paired spots.
  - Active fraction 4–42 % across datasets (mean 16 %).
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from bessel_seg.config import TemporalValidationConfig
from bessel_seg.data_types import SpotPair

logger = logging.getLogger(__name__)

# Pixel radius for circular ROI extraction around the cluster centre
_ROI_RADIUS: int = 6


def _extract_trace(
    dff: NDArray[np.float32],
    cy: float,
    cx: float,
    radius: int = _ROI_RADIUS,
) -> NDArray[np.float32]:
    """Extract mean ΔF/F₀ trace from a circular ROI.

    Args:
        dff: (T, H, W) ΔF/F₀ stack.
        cy, cx: Centre row and column of the ROI.
        radius: Pixel radius of the circular ROI.

    Returns:
        (T,) float32 trace.
    """
    T, H, W = dff.shape
    iy, ix = int(round(cy)), int(round(cx))

    # Build coordinate grid once and apply circular mask
    ys = np.arange(max(0, iy - radius), min(H, iy + radius + 1))
    xs = np.arange(max(0, ix - radius), min(W, ix + radius + 1))
    if ys.size == 0 or xs.size == 0:
        return np.zeros(T, dtype=np.float32)

    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    in_circle = (yg - iy) ** 2 + (xg - ix) ** 2 <= radius ** 2

    patch = dff[:, ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]  # (T, ph, pw)
    # Mask out pixels outside circle for each frame
    mask_2d = in_circle  # (ph, pw)
    n_pixels = int(mask_2d.sum())
    if n_pixels == 0:
        return np.zeros(T, dtype=np.float32)

    trace = patch[:, mask_2d].mean(axis=1).astype(np.float32)
    return trace


def _cluster_centre(cluster: list[SpotPair]) -> tuple[float, float]:
    """Compute the mean (y, x) centre of a cluster.

    Args:
        cluster: Non-empty list of SpotPair objects.

    Returns:
        (cy, cx) mean centre.
    """
    centres = np.array([p.center for p in cluster], dtype=np.float64)
    cy, cx = centres.mean(axis=0)
    return float(cy), float(cx)


def _find_events(
    trace: NDArray[np.float32],
    config: TemporalValidationConfig,
) -> list[tuple[int, int, float]]:
    """Detect calcium transient events in a ΔF/F₀ trace.

    An event is a contiguous run of frames where the trace exceeds
    ``min_peak_dff * 0.5`` (half-peak threshold), anchored around a local
    maximum that itself exceeds ``min_peak_dff``.

    For each candidate event the function also verifies:
      - At least ``min_rise_frames`` frames with increasing signal before the
        peak frame.
      - At least ``min_decay_frames`` frames with decreasing signal after the
        peak frame.
      - The contiguous above-threshold run is ≤ ``max_event_duration`` frames.

    Args:
        trace: (T,) ΔF/F₀ trace.
        config: :class:`~bessel_seg.config.TemporalValidationConfig`.

    Returns:
        List of ``(start, end, peak_dff)`` tuples for accepted events.
        ``start`` and ``end`` are inclusive frame indices.
    """
    T = len(trace)
    peak_threshold = config.min_peak_dff
    half_threshold = peak_threshold * 0.5

    # Find frames above half-peak threshold to define event regions
    above = trace >= half_threshold  # (T,) bool

    # Label contiguous runs above threshold
    events: list[tuple[int, int, float]] = []
    t = 0
    while t < T:
        if not above[t]:
            t += 1
            continue
        # Found start of a run
        start = t
        while t < T and above[t]:
            t += 1
        end = t - 1  # inclusive

        # Peak within this run
        peak_val = float(trace[start : end + 1].max())
        if peak_val < peak_threshold:
            continue  # local max below threshold

        peak_idx = int(start + np.argmax(trace[start : end + 1]))

        # Duration gate
        duration = end - start + 1
        if duration > config.max_event_duration:
            continue

        # Rise check: at least min_rise_frames of strictly increasing signal
        # before the peak (look back from peak_idx)
        rise_frames = 0
        for k in range(peak_idx - 1, max(start - 1, -1), -1):
            if trace[k] < trace[k + 1]:
                rise_frames += 1
            else:
                break
        if rise_frames < config.min_rise_frames:
            # Looser check: any frame before the peak that is lower counts
            pre_peak = trace[start:peak_idx]
            if pre_peak.size > 0 and float(pre_peak.min()) < peak_val:
                rise_frames = max(rise_frames, 1)
            if rise_frames < config.min_rise_frames:
                continue

        # Decay check: at least min_decay_frames of strictly decreasing signal
        # after the peak
        decay_frames = 0
        for k in range(peak_idx + 1, min(end + 1, T)):
            if trace[k] < trace[k - 1]:
                decay_frames += 1
            else:
                break
        if decay_frames < config.min_decay_frames:
            # Looser check: any frame after the peak that is lower counts
            post_peak = trace[peak_idx + 1 : end + 1]
            if post_peak.size > 0 and float(post_peak.min()) < peak_val:
                decay_frames = max(decay_frames, 1)
            if decay_frames < config.min_decay_frames:
                continue

        events.append((start, end, peak_val))

    return events


def validate_calcium_dynamics(
    clusters: list[list[SpotPair]],
    dff: NDArray[np.float32],
    config: TemporalValidationConfig,
) -> list[list[SpotPair]]:
    """Filter neuron candidate clusters by temporal calcium dynamics.

    For each cluster:

    1. Compute the weighted mean centre across all SpotPairs in the cluster.
    2. Extract a circular ROI (radius ``_ROI_RADIUS`` = 6 px) ΔF/F₀ trace
       from ``dff``.
    3. **Reject** if ``active_fraction > max_active_fraction`` — persistent
       bright region is background, not a neuron.
    4. **Reject** if no valid calcium-transient event is detected
       (peak ΔF/F₀ ≥ 0.20, rise-peak-decay structure).
    5. **Accept** otherwise.

    Args:
        clusters: Output of
            :func:`~bessel_seg.fusion.spatial_cluster.cluster_neuron_detections`.
        dff: (T, H, W) ΔF/F₀ stack.
        config: :class:`~bessel_seg.config.TemporalValidationConfig`.

    Returns:
        Subset of *clusters* that pass temporal validation, in the same
        order as the input.
    """
    T = dff.shape[0]
    validated: list[list[SpotPair]] = []

    for cluster_idx, cluster in enumerate(clusters):
        cy, cx = _cluster_centre(cluster)

        # For paired neurons the cluster centre lies midway between the two spots
        # (spacing ~43 px >> 3σ ≈ 7.5 px), so the centre position has little to no
        # dFF signal.  To handle both paired neurons (signal at spot positions) and
        # the degenerate case where signal is centred at the midpoint, we extract
        # traces at the cluster centre AND at every individual spot position, then
        # take the element-wise maximum across all positions.
        candidate_positions: list[tuple[float, float]] = [(cy, cx)]  # centre first
        for p in cluster:
            candidate_positions.append((p.left.y, p.left.x))
            candidate_positions.append((p.right.y, p.right.x))

        # Deduplicate positions within _ROI_RADIUS of each other
        unique_positions: list[tuple[float, float]] = []
        for pos in candidate_positions:
            if not any(
                math.hypot(pos[0] - up[0], pos[1] - up[1]) < _ROI_RADIUS
                for up in unique_positions
            ):
                unique_positions.append(pos)

        all_traces = np.stack(
            [_extract_trace(dff, py, px) for py, px in unique_positions], axis=0
        )
        trace = all_traces.max(axis=0)  # (T,) element-wise max across positions

        # --- Gate 1: Persistent background rejection ---
        peak_dff = float(trace.max())
        if peak_dff < config.min_peak_dff:
            logger.debug(
                "Cluster %d @ (%.1f, %.1f) rejected: peak_dff=%.3f < %.3f",
                cluster_idx, cy, cx, peak_dff, config.min_peak_dff,
            )
            continue

        half_peak = peak_dff * 0.5
        active_fraction = float((trace >= half_peak).mean())
        if active_fraction >= config.max_active_fraction:
            logger.debug(
                "Cluster %d @ (%.1f, %.1f) rejected: active_frac=%.3f >= %.3f",
                cluster_idx, cy, cx, active_fraction, config.max_active_fraction,
            )
            continue

        # --- Gate 2: Rise-peak-decay event detection ---
        events = _find_events(trace, config)
        if not events:
            logger.debug(
                "Cluster %d @ (%.1f, %.1f) rejected: no valid transient events",
                cluster_idx, cy, cx,
            )
            continue

        logger.debug(
            "Cluster %d @ (%.1f, %.1f) accepted: %d events, peak_dff=%.3f, "
            "active_frac=%.3f",
            cluster_idx, cy, cx, len(events), peak_dff, active_fraction,
        )
        validated.append(cluster)

    logger.info(
        "validate_calcium_dynamics: %d clusters → %d validated (%.0f%%)",
        len(clusters),
        len(validated),
        100.0 * len(validated) / max(len(clusters), 1),
    )
    return validated
