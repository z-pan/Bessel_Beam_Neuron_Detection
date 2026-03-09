"""Temporal summary statistics maps for the Enhanced Detection Image (EDI).

Each map condenses the (T, H, W) stack to a single (H, W) image capturing a
different aspect of temporal activity.  All maps are normalised to [0, 1]
before being returned so they can be fused with equal weight in the EDI.

Maps produced:
  ``"sigma"``    — temporal std of ΔF/F₀; highlights pixels that fluctuate.
  ``"max_proj"`` — temporal maximum of ΔF/F₀; catches peak transients.
  ``"mean"``     — temporal mean of the raw stack; shows overall brightness.
  ``"cv"``       — coefficient of variation = std(raw) / (mean(raw) + ε);
                   relative variability, useful for dim neurons.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Small constant added to denominators to avoid division-by-zero
_EPS: float = 1e-9


def _normalise_to_01(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Linearly rescale *arr* to [0, 1].

    If all values are identical (range == 0) the array is returned as
    all-zeros rather than raising an error.
    """
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < _EPS:
        return np.zeros_like(arr)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def compute_summary_maps(
    stack: NDArray[np.float32],
    dff: NDArray[np.float32],
) -> dict[str, NDArray[np.float32]]:
    """Compute temporal summary statistics maps from the imaging stack.

    Args:
        stack: (T, H, W) float32 raw (or photobleach-corrected) intensity
            stack.  Used for the ``"mean"`` and ``"cv"`` maps.
        dff:   (T, H, W) float32 ΔF/F₀ stack from
            :func:`~bessel_seg.temporal.delta_f.compute_delta_f`.
            Used for the ``"sigma"`` and ``"max_proj"`` maps.

    Returns:
        Dictionary with keys ``"sigma"``, ``"max_proj"``, ``"mean"``,
        ``"cv"``, each mapping to a (H, W) float32 array normalised to
        [0, 1].

    Notes:
        All computations are along ``axis=0`` (the time axis).
        Normalisation uses global min–max so maps are comparable within a
        single dataset run but not across datasets.
    """
    # --- ΔF/F₀ -based maps --------------------------------------------------
    # Temporal std of ΔF/F₀ — primary indicator of activity
    sigma_map: NDArray[np.float32] = np.std(dff, axis=0).astype(np.float32)

    # Temporal maximum of ΔF/F₀ — captures peak transient amplitude
    max_proj: NDArray[np.float32] = np.max(dff, axis=0).astype(np.float32)

    # --- Raw stack maps -------------------------------------------------------
    raw_mean: NDArray[np.float32] = np.mean(stack, axis=0).astype(np.float32)

    # CV = std / (mean + ε) — relative variability per pixel
    raw_std: NDArray[np.float32] = np.std(stack, axis=0).astype(np.float32)
    cv: NDArray[np.float32] = (raw_std / (raw_mean + _EPS)).astype(np.float32)

    # --- Normalise each map to [0, 1] ----------------------------------------
    maps = {
        "sigma":    _normalise_to_01(sigma_map),
        "max_proj": _normalise_to_01(max_proj),
        "mean":     _normalise_to_01(raw_mean),
        "cv":       _normalise_to_01(cv),
    }

    logger.info(
        "Summary maps computed (all normalised to [0,1]): "
        "sigma_max=%.4f  max_proj_max=%.4f  mean_max=%.4f  cv_max=%.4f",
        float(maps["sigma"].max()),
        float(maps["max_proj"].max()),
        float(maps["mean"].max()),
        float(maps["cv"].max()),
    )
    return maps
