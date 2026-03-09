"""Adaptive local-SNR filtering for detected spot candidates.

Each candidate spot is evaluated against its local neighbourhood:

    Inner region : circle of radius 2σ around the spot centre
    Outer annulus: ring from radius 2σ to 4σ

    local_SNR = (mean_inner − mean_outer) / (std_outer + ε)

Spots whose local SNR falls below ``config.local_snr_threshold`` (calibrated
from real data: paired-neuron p10 SNR = 0.41) are discarded as noise.
Spots whose centre is within ``config.max_sigma`` pixels of the image border
are also removed to avoid truncated profiles.

Calibration note (CLAUDE.md, 3 346 Gaussian fits):
  - Paired neuron SNR: median=1.47, p10=0.41 → threshold=0.4
  - Single-spot SNR:   median=1.03, p10=0.31 (lower threshold not applied here)
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from bessel_seg.config import DetectionConfig
from bessel_seg.data_types import Spot

logger = logging.getLogger(__name__)

# Prevents division by near-zero outer-ring std
_STD_EPS: float = 1e-6

# Annulus multipliers: inner radius = INNER_MULT * sigma, outer = OUTER_MULT * sigma
_INNER_MULT: float = 2.0
_OUTER_MULT: float = 4.0


def _compute_local_snr(
    image: NDArray[np.float32],
    cy: float,
    cx: float,
    sigma: float,
) -> float:
    """Compute the local SNR of a spot using an inner-disc / outer-annulus model.

    Args:
        image: (H, W) float32 image.
        cy, cx: Sub-pixel spot centre (row, column).
        sigma: Gaussian sigma of the spot (from LoG detection).

    Returns:
        local_snr: ``(mean_inner − mean_outer) / (std_outer + ε)``.
        Returns 0.0 if the inner region contains no pixels.
    """
    H, W = image.shape
    r_inner = _INNER_MULT * sigma
    r_outer = _OUTER_MULT * sigma

    # Build a coordinate grid over the bounding box of the outer circle
    row_lo = max(0, int(cy - r_outer) - 1)
    row_hi = min(H, int(cy + r_outer) + 2)
    col_lo = max(0, int(cx - r_outer) - 1)
    col_hi = min(W, int(cx + r_outer) + 2)

    rows = np.arange(row_lo, row_hi, dtype=np.float64)
    cols = np.arange(col_lo, col_hi, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")  # (patch_H, patch_W)

    dist2 = (rr - cy) ** 2 + (cc - cx) ** 2
    r_inner2 = r_inner ** 2
    r_outer2 = r_outer ** 2

    inner_mask = dist2 <= r_inner2
    outer_mask = (dist2 > r_inner2) & (dist2 <= r_outer2)

    patch = image[row_lo:row_hi, col_lo:col_hi].astype(np.float64)

    inner_pixels = patch[inner_mask]
    outer_pixels = patch[outer_mask]

    if inner_pixels.size == 0 or outer_pixels.size == 0:
        return 0.0

    mean_inner = float(inner_pixels.mean())
    mean_outer = float(outer_pixels.mean())
    std_outer  = float(outer_pixels.std())

    snr = (mean_inner - mean_outer) / (std_outer + _STD_EPS)
    return snr


def filter_spots_adaptive(
    spots: list[Spot],
    image: NDArray[np.float32],
    config: DetectionConfig,
) -> list[Spot]:
    """Filter spot candidates by local SNR and border proximity.

    Two rejection criteria applied in order:

    1. **Border rejection**: discard any spot whose centre is within
       ``config.max_sigma`` pixels of the image edge (truncated PSF).

    2. **Local SNR rejection**: compute the inner-disc / outer-annulus SNR for
       each remaining spot and discard those below
       ``config.local_snr_threshold`` (default 0.4).

    The accepted spots have their ``snr`` field updated with the measured value.

    Args:
        spots: Candidate spots from :func:`~bessel_seg.detection.blob_detect.detect_spots`.
        image: (H, W) float32 image used for SNR measurement (same image that
            was passed to :func:`detect_spots`).
        config: Detection configuration (provides ``local_snr_threshold`` and
            ``max_sigma`` for border exclusion).

    Returns:
        Filtered list of :class:`~bessel_seg.data_types.Spot` objects with
        ``snr`` populated.
    """
    if not spots:
        return []

    H, W = image.shape
    border = config.max_sigma  # pixels to exclude near each edge
    accepted: list[Spot] = []
    n_border = 0
    n_snr = 0

    for spot in spots:
        # --- Criterion 1: border proximity ---
        if (
            spot.y < border
            or spot.y > H - 1 - border
            or spot.x < border
            or spot.x > W - 1 - border
        ):
            n_border += 1
            continue

        # --- Criterion 2: local SNR ---
        snr = _compute_local_snr(image, spot.y, spot.x, spot.sigma)
        if snr < config.local_snr_threshold:
            n_snr += 1
            continue

        spot.snr = snr
        accepted.append(spot)

    logger.debug(
        "filter_spots_adaptive: %d in → %d out  "
        "(border=%d, low_snr=%d, threshold=%.2f)",
        len(spots), len(accepted),
        n_border, n_snr, config.local_snr_threshold,
    )
    return accepted
