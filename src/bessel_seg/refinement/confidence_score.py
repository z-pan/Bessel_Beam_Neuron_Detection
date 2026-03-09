"""Confidence scoring for neuron ROIs — four-tier model.

CLAUDE.md defines four confidence tiers based on detection path and temporal
signal quality:

+--------------------------------------+-------------+
| Scenario                             | Range       |
+======================================+=============+
| Paired + temporal correlation        | 0.80 – 1.00 |
+--------------------------------------+-------------+
| Single-spot + near known pair        | 0.60 – 0.80 |
+--------------------------------------+-------------+
| Single-spot + good temporal dynamics | 0.40 – 0.60 |
+--------------------------------------+-------------+
| Single-spot + weak dynamics          | 0.20 – 0.40 |
+--------------------------------------+-------------+

Within each tier the score is modulated by signal-quality metrics that are
computed from the temporal ΔF/F₀ trace and (for paired clusters) from
geometric quality of the matched pairs.

Public API
----------
:func:`compute_confidence` — main entry point.  Accepts either a
``list[SpotPair]`` (paired cluster) or a
:class:`~bessel_seg.data_types.NeuronROI` (single-spot neuron) as its first
argument and returns a float in [0, 1].
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from bessel_seg.data_types import FrameQuality, NeuronROI, SpotPair

logger = logging.getLogger(__name__)

# ΔF/F₀ value that saturates the temporal quality score (peak_dff ≥ this → 1.0)
_PEAK_SATURATE: float = 2.0
# Active fraction at which temporal quality reaches zero (persistent background)
_MAX_ACTIVE_FRAC: float = 0.50
# Minimum peak_dff considered as any signal (below → temporal quality = 0)
_MIN_PEAK_DFF: float = 0.20


def _temporal_quality(trace: NDArray[np.float32]) -> float:
    """Score the temporal trace quality in [0, 1].

    Two sub-scores, equally weighted:

    * **peak score**: ``min(peak_dff / _PEAK_SATURATE, 1.0)`` — rewards
      high ΔF/F₀ amplitude.
    * **transient score**: ``1 − min(active_frac / _MAX_ACTIVE_FRAC, 1.0)``
      — rewards brief, transient signals (penalises persistent background).

    Args:
        trace: (T,) ΔF/F₀ trace.

    Returns:
        Quality score in [0, 1].  Returns 0.0 for empty or flat traces.
    """
    if trace.size == 0:
        return 0.0

    peak = float(trace.max())
    if peak < _MIN_PEAK_DFF:
        return 0.0

    peak_score = min(peak / _PEAK_SATURATE, 1.0)

    half_peak = peak * 0.5
    active_frac = float((trace >= half_peak).mean())
    transient_score = max(1.0 - active_frac / _MAX_ACTIVE_FRAC, 0.0)

    return float((peak_score + transient_score) / 2.0)


def _paired_cluster_raw_quality(cluster: list[SpotPair], total_frames: int) -> float:
    """Compute a raw quality score in [0, 1] for a paired cluster.

    Components (equal weight = 0.25 each):

    * **detection_fraction**: ``len(cluster) / total_frames`` (capped at 1).
    * **pair_quality**: mean of ``(intensity_ratio + scale_similarity) / 2``
      across all pairs.  Both metrics are in [0, 1] by construction.
    * **pair_cost_quality**: ``1 − mean(pair_cost)`` clipped to [0, 1].
      Lower Hungarian cost → higher quality.
    * **spatial_consistency**: ``1 / (1 + σ_centres)`` where σ_centres is
      the sum of position standard deviations in y and x across the cluster.

    Args:
        cluster: Non-empty list of SpotPair objects.
        total_frames: Total frames in the recording.

    Returns:
        Raw quality score in [0, 1].
    """
    if not cluster:
        return 0.0

    # 1. Detection fraction
    det_frac = min(len(cluster) / max(total_frames, 1), 1.0)

    # 2. Pair quality (per-pair geometry scores, already normalised to [0,1])
    pair_qual = float(
        np.mean([(p.intensity_ratio + p.scale_similarity) / 2.0 for p in cluster])
    )

    # 3. Pair cost quality (lower cost → higher quality)
    mean_cost = float(np.mean([p.pair_cost for p in cluster]))
    cost_qual = float(np.clip(1.0 - mean_cost, 0.0, 1.0))

    # 4. Spatial consistency: neurons are spatially stable; tight cluster → high score
    centres = np.array([p.center for p in cluster], dtype=np.float64)  # (N, 2)
    std_y = float(np.std(centres[:, 0]))
    std_x = float(np.std(centres[:, 1]))
    spatial_cons = 1.0 / (1.0 + std_y + std_x)

    raw = (det_frac + pair_qual + cost_qual + spatial_cons) / 4.0
    return float(np.clip(raw, 0.0, 1.0))


def compute_confidence(
    cluster_or_single: Union[list[SpotPair], NeuronROI],
    frame_qualities: list[FrameQuality],
    temporal_trace: NDArray[np.float32],
    total_frames: int,
) -> float:
    """Compute the overall confidence score for a neuron ROI.

    The score is placed in the tier appropriate for the detection path and
    then modulated upward within that tier by signal-quality metrics.

    **Paired path** (``cluster_or_single`` is a ``list[SpotPair]``):
        - Tier base: **0.80**; ceiling: **1.00**.
        - Modulation: equally-weighted combination of detection fraction,
          pair geometry quality, pair-cost quality, and spatial consistency,
          then scaled by temporal quality.
        - Formula: ``0.80 + 0.20 · cluster_quality · temporal_quality``

    **Single-spot path** (``cluster_or_single`` is a ``NeuronROI``):
        - ``"single_near_pair"`` → tier **[0.60, 0.80]**
        - ``"single_isolated"`` → tier **[0.40, 0.60]**
        - Any other / fallback → tier **[0.20, 0.40]**
        - Modulation: temporal quality fills the 0.20-wide tier window.
        - Formula: ``tier_base + 0.20 · temporal_quality``

    Args:
        cluster_or_single: Either a non-empty ``list[SpotPair]`` from
            spatial clustering (paired path) or a
            :class:`~bessel_seg.data_types.NeuronROI` from the single-spot
            path.
        frame_qualities: Per-frame quality objects (used for future
            extensions; currently unused in the formula but accepted for
            API consistency).
        temporal_trace: (T,) ΔF/F₀ trace at the neuron location.
        total_frames: Total frames in the recording (denominator for
            detection fraction in the paired path).

    Returns:
        Confidence score clipped to [0, 1].
    """
    tq = _temporal_quality(temporal_trace)

    if isinstance(cluster_or_single, list):
        # ----------------------------------------------------------------
        # Paired path: tier [0.80, 1.00]
        # ----------------------------------------------------------------
        cluster: list[SpotPair] = cluster_or_single
        raw_qual = _paired_cluster_raw_quality(cluster, total_frames)
        # Both cluster quality and temporal quality must be good for max score
        combined = (raw_qual + tq) / 2.0
        score = 0.80 + 0.20 * combined
        logger.debug(
            "compute_confidence [paired]: raw_qual=%.3f tq=%.3f → %.3f",
            raw_qual, tq, score,
        )
    else:
        # ----------------------------------------------------------------
        # Single-spot path: tier determined by detection_type
        # ----------------------------------------------------------------
        roi: NeuronROI = cluster_or_single
        det_type = roi.detection_type

        if det_type == "single_near_pair":
            tier_base = 0.60   # range [0.60, 0.80]
        elif det_type == "single_isolated":
            tier_base = 0.40   # range [0.40, 0.60]
        else:
            tier_base = 0.20   # range [0.20, 0.40] — weak / unknown

        score = tier_base + 0.20 * tq
        logger.debug(
            "compute_confidence [%s]: tq=%.3f tier_base=%.2f → %.3f",
            det_type, tq, tier_base, score,
        )

    return float(np.clip(score, 0.0, 1.0))
