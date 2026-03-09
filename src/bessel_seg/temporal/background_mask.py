"""Illumination masking and band-artifact suppression.

IMPORTANT — Original approach retired:
    The original CV / presence-ratio background mask returned ALL ZEROS on
    every real dataset because the detection criteria (high mean + low CV +
    high presence) are too strict for 8-bit data where most pixels are near
    zero.  This module implements the redesigned approach described in
    CLAUDE.md.

Redesigned approach:
  1. ``generate_illumination_mask`` — threshold on temporal mean to find the
     valid illuminated FOV region; exclude permanently dark areas (outside
     illumination cone, corners, dark bottom strip).
  2. ``suppress_band_artifacts`` — subtract a smooth row-profile from the
     image to remove the horizontal banding caused by Bessel beam scanning.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_opening, gaussian_filter1d
from skimage.morphology import disk

logger = logging.getLogger(__name__)


def generate_illumination_mask(
    stack: NDArray[np.float32],
    threshold_percentile: int = 5,
) -> NDArray[np.bool_]:
    """Identify the valid illuminated region of the field of view.

    Pixels whose temporal mean falls below the *threshold_percentile*-th
    percentile of the overall mean image are considered outside the
    illumination cone (permanently dark) and are excluded from detection.

    Real data characteristics driving this design:
      - Bottom portion of the FOV is often completely dark.
      - Non-uniformity across the FOV ranges from 2.5× to 14.6×.
      - The original CV/presence mask returned all zeros — this replaces it.

    Args:
        stack: (T, H, W) float32 image stack.
        threshold_percentile: Pixels with temporal mean below this percentile
            of the mean image are marked as *outside* the FOV (mask = False).
            Default 5 retains almost all illuminated tissue while safely
            excluding truly dark regions.

    Returns:
        mask: (H, W) boolean array.  ``True`` = valid illuminated pixel.
    """
    # Temporal mean image — (H, W)
    mean_img: NDArray[np.float32] = np.mean(stack, axis=0).astype(np.float32)

    # Threshold: use the specified percentile of the mean image itself,
    # but always require at least 1.0 so zero-background pixels are excluded.
    threshold = float(np.percentile(mean_img, threshold_percentile))
    threshold = max(threshold, 1.0)

    raw_mask: NDArray[np.bool_] = mean_img > threshold

    # Morphological opening removes small isolated bright specs and cleans
    # the boundary of the illuminated region.
    struct = disk(5)  # radius-5 disk structuring element
    mask: NDArray[np.bool_] = binary_opening(raw_mask, structure=struct)

    n_valid = int(mask.sum())
    n_total = mask.size
    logger.info(
        "Illumination mask: threshold=%.3f (p%d of mean image), "
        "valid pixels=%d / %d (%.1f%%)",
        threshold,
        threshold_percentile,
        n_valid,
        n_total,
        100.0 * n_valid / n_total,
    )
    if n_valid == 0:
        logger.warning(
            "Illumination mask is all-False!  All pixels were excluded.  "
            "Consider lowering threshold_percentile."
        )

    return mask.astype(np.bool_)


def suppress_band_artifacts(
    image: NDArray[np.float32],
    orientation: str = "horizontal",
) -> NDArray[np.float32]:
    """Suppress stripe / band artifacts from the EDI or summary map.

    The Bessel beam scanning pattern produces strong horizontal bands in the
    EDI.  These appear as alternating bright/dark rows and generate false
    positive spot detections along band boundaries.

    The correction subtracts a smoothed version of the row-averaged intensity
    profile, effectively acting as a row-wise high-pass filter.

    Args:
        image: (H, W) float32 input map (typically the EDI).
        orientation: ``"horizontal"`` — correct row-wise bands (default, matches
            Bessel beam scanning direction).  Other values raise ``ValueError``.

    Returns:
        Corrected (H, W) float32 image with negative values clipped to zero.

    Raises:
        ValueError: If *orientation* is not ``"horizontal"``.
    """
    if orientation != "horizontal":
        raise ValueError(
            f"Only 'horizontal' orientation is supported, got '{orientation}'."
        )

    # Row-wise mean profile: average each row across all columns → (H,)
    row_profile: NDArray[np.float64] = np.mean(image, axis=1).astype(np.float64)

    # Smooth the profile to capture the slow band structure while leaving
    # the sharp spot signals intact.  sigma=20 matches ~40-px band width.
    row_smooth: NDArray[np.float64] = gaussian_filter1d(row_profile, sigma=20)

    # Subtract the smooth background from each column in the image
    corrected = image - row_smooth[:, np.newaxis].astype(np.float32)

    # Clip negative values to zero (negative signal is not meaningful here)
    corrected = np.clip(corrected, 0.0, None).astype(np.float32)

    logger.debug(
        "Band artifact suppression: removed row profile range [%.4f, %.4f]; "
        "post-correction range [%.4f, %.4f]",
        float(row_smooth.min()),
        float(row_smooth.max()),
        float(corrected.min()),
        float(corrected.max()),
    )
    return corrected
