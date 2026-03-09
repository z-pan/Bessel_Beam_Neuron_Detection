"""Build the Enhanced Detection Image (EDI) from temporal summary maps.

The EDI fuses three complementary summary statistics into a single (H, W) map
that highlights pixels most likely to contain neuron spots:

    EDI = α · sigma_map + β · max_proj_map + γ · cv_map

Default weights (from EDIConfig): α=0.4, β=0.4, γ=0.2.

After fusion the EDI is processed to remove horizontal band artifacts caused by
the Bessel beam scanning pattern, masked outside the illumination region, and
normalised to [0, 1].
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from bessel_seg.config import EDIConfig
from bessel_seg.temporal.background_mask import suppress_band_artifacts

logger = logging.getLogger(__name__)

# Minimum denominator used when normalising the final EDI to [0, 1]
_NORM_EPS: float = 1e-9


def build_enhanced_detection_image(
    summary_maps: dict[str, NDArray[np.float32]],
    illumination_mask: NDArray[np.bool_],
    config: EDIConfig,
) -> NDArray[np.float32]:
    """Build the Enhanced Detection Image (EDI) for downstream spot detection.

    Each input summary map must already be normalised to [0, 1] (as returned by
    :func:`~bessel_seg.temporal.summary_maps.compute_summary_maps`).

    Pipeline:
        1. Weighted sum of ``sigma``, ``max_proj``, and ``cv`` maps.
        2. Suppress horizontal band artifacts via
           :func:`~bessel_seg.temporal.background_mask.suppress_band_artifacts`.
        3. Zero-out pixels outside the illumination region.
        4. Normalise the result to [0, 1].

    Args:
        summary_maps: Dict with at least keys ``"sigma"``, ``"max_proj"``, and
            ``"cv"``.  Each value is a (H, W) float32 array in [0, 1].
        illumination_mask: (H, W) boolean array; ``True`` = valid illuminated
            pixel.  Pixels where the mask is ``False`` are set to 0 in the EDI.
        config: :class:`~bessel_seg.config.EDIConfig` carrying the fusion weights
            ``sigma_weight``, ``maxproj_weight``, and ``corr_weight``.

    Returns:
        edi: (H, W) float32 array normalised to [0, 1].

    Raises:
        KeyError: If required keys are missing from *summary_maps*.
    """
    for key in ("sigma", "max_proj", "cv"):
        if key not in summary_maps:
            raise KeyError(
                f"Required summary map '{key}' not found in summary_maps. "
                f"Available keys: {list(summary_maps.keys())}"
            )

    sigma_map = summary_maps["sigma"]      # (H, W) in [0, 1]
    max_proj  = summary_maps["max_proj"]   # (H, W) in [0, 1]
    cv_map    = summary_maps["cv"]         # (H, W) in [0, 1]

    # Step 1 — weighted fusion
    edi: NDArray[np.float32] = (
        config.sigma_weight    * sigma_map
        + config.maxproj_weight * max_proj
        + config.corr_weight    * cv_map
    ).astype(np.float32)

    logger.debug(
        "EDI after fusion: min=%.4f, max=%.4f  "
        "(weights: σ=%.2f, max=%.2f, cv=%.2f)",
        float(edi.min()), float(edi.max()),
        config.sigma_weight, config.maxproj_weight, config.corr_weight,
    )

    # Step 2 — suppress horizontal band artifacts from the Bessel beam scan
    edi = suppress_band_artifacts(edi, orientation="horizontal")

    # Step 3 — mask outside the illumination cone
    edi[~illumination_mask] = 0.0

    # Step 4 — normalise to [0, 1]
    lo = float(edi.min())
    hi = float(edi.max())
    if hi - lo > _NORM_EPS:
        edi = ((edi - lo) / (hi - lo)).astype(np.float32)
    else:
        # Flat or all-zero EDI (e.g., all-dark frame or fully masked)
        logger.warning(
            "EDI has near-zero range (max-min=%.2e); returning zeros.", hi - lo
        )
        edi = np.zeros_like(edi)

    logger.info(
        "EDI built: shape=%s, non-zero pixels=%d / %d",
        edi.shape,
        int((edi > 0).sum()),
        edi.size,
    )
    return edi
