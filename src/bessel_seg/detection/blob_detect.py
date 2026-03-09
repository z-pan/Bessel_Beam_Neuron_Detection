"""Multi-scale Laplacian-of-Gaussian (LoG) spot detection.

Two entry points:

* :func:`detect_spots` — detect spots in a single (H, W) image (EDI or dff
  frame).  Returns a list of :class:`~bessel_seg.data_types.Spot` objects.

* :func:`detect_spots_per_frame` — run :func:`detect_spots` independently on
  every frame of a (T, H, W) ΔF/F₀ stack.  Frames with no signal above a
  quality threshold are skipped.

Calibrated detection parameters (from 3 346 Gaussian fits on real neurons):
  - Spot sigma range: p10=1.22 px, p90=4.28 px
  - Config defaults: min_sigma=1.0, max_sigma=5.0, threshold=0.03
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from skimage.feature import blob_log

from bessel_seg.config import DetectionConfig
from bessel_seg.data_types import Spot

logger = logging.getLogger(__name__)

# Frames whose peak ΔF/F₀ is below this value are skipped (no signal)
_DEFAULT_QUALITY_THRESHOLD: float = 0.1


def detect_spots(
    image: NDArray[np.float32],
    config: DetectionConfig,
) -> list[Spot]:
    """Detect candidate fluorescent spots using multi-scale LoG.

    Args:
        image: (H, W) float32 image (EDI or single ΔF/F₀ frame), values in
            [0, 1] or in ΔF/F₀ units.  The image should already have band
            artifacts suppressed and the illumination mask applied.
        config: :class:`~bessel_seg.config.DetectionConfig` with LoG parameters.

    Returns:
        List of :class:`~bessel_seg.data_types.Spot` objects, one per detected
        blob.  ``frame_idx`` is ``None`` (caller sets it if needed).
    """
    if image.size == 0:
        return []

    # skimage.feature.blob_log returns an (N, 3) array: [y, x, sigma]
    # sigma here is the Gaussian sigma *scaled by sqrt(2)* internally, but the
    # returned value is the LoG sigma — consistent with the config units.
    blobs = blob_log(
        image,
        min_sigma=config.min_sigma,
        max_sigma=config.max_sigma,
        num_sigma=config.num_sigma,
        threshold=config.threshold,
        overlap=config.overlap,
    )  # shape: (N, 3)

    spots: list[Spot] = []
    H, W = image.shape
    for row in blobs:
        y, x, sigma = float(row[0]), float(row[1]), float(row[2])
        # Peak intensity at the blob centre (nearest integer coordinates)
        iy = int(round(y))
        ix = int(round(x))
        # Clamp to valid pixel range
        iy = max(0, min(iy, H - 1))
        ix = max(0, min(ix, W - 1))
        intensity = float(image[iy, ix])
        spots.append(Spot(y=y, x=x, sigma=sigma, intensity=intensity))

    logger.debug(
        "detect_spots: found %d blobs in image of shape %s "
        "(min_sigma=%.1f, max_sigma=%.1f, threshold=%.4f)",
        len(spots), image.shape,
        config.min_sigma, config.max_sigma, config.threshold,
    )
    return spots


def detect_spots_per_frame(
    dff_stack: NDArray[np.float32],
    illumination_mask: NDArray[np.bool_],
    config: DetectionConfig,
    quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
) -> dict[int, list[Spot]]:
    """Run spot detection independently on each frame of a ΔF/F₀ stack.

    Frames with little signal (``max(dff[t]) < quality_threshold``) are skipped
    entirely to avoid accumulating false positives on noise frames.

    Args:
        dff_stack: (T, H, W) float32 ΔF/F₀ stack.
        illumination_mask: (H, W) boolean mask; pixels outside the illuminated
            FOV are zeroed before detection.
        config: Detection parameters.
        quality_threshold: Frames whose maximum ΔF/F₀ is below this value are
            skipped.  Default 0.1 (well below the 0.20 ΔF/F₀ detection floor).

    Returns:
        Dict mapping ``frame_idx`` → list of :class:`~bessel_seg.data_types.Spot`.
        Only frames that passed the quality check and contained ≥ 1 spot are
        included.
    """
    T = dff_stack.shape[0]
    results: dict[int, list[Spot]] = {}
    skipped = 0

    for t in range(T):
        frame = dff_stack[t].copy()  # (H, W)

        # Quality gate: skip low-signal frames
        if float(np.max(frame)) < quality_threshold:
            skipped += 1
            continue

        # Zero out outside-illumination pixels before detection
        frame[~illumination_mask] = 0.0

        spots = detect_spots(frame, config)

        # Tag each spot with its source frame index
        for s in spots:
            s.frame_idx = t

        if spots:
            results[t] = spots

    logger.info(
        "detect_spots_per_frame: processed %d frames, skipped %d (max dff < %.2f), "
        "found spots in %d frames",
        T, skipped, quality_threshold, len(results),
    )
    return results
