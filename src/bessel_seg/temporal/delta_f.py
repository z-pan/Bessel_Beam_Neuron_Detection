"""ΔF/F₀ normalisation of calcium imaging stacks.

Converts raw intensity values into fractional fluorescence changes relative
to the per-pixel baseline.  The *epsilon* floor prevents division by
near-zero baseline values (which can occur in dark or out-of-illumination
pixels even after baseline clamping).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_delta_f(
    stack: NDArray[np.float32],
    baseline: NDArray[np.float32],
    epsilon: float = 1.0,
) -> NDArray[np.float32]:
    """Compute the ΔF/F₀ normalised stack.

    .. math::

        \\Delta F / F_0(t, y, x) =
            \\frac{\\text{stack}(t, y, x) - F_0(y, x)}
                 {\\max(F_0(y, x),\\, \\epsilon)}

    Measured distributions on real data:
      - Paired neurons: median ΔF/F₀ ≈ 0.76, p10 ≈ 0.21
      - Single-spot neurons: median ΔF/F₀ ≈ 0.54, p10 ≈ 0.13

    Args:
        stack:    (T, H, W) float32 raw or photobleach-corrected intensity.
        baseline: (H, W) float32 per-pixel baseline F₀.  Should already be
            clamped to ≥ 1.0 by :func:`~bessel_seg.temporal.baseline.estimate_baseline`.
        epsilon:  Safety floor added to the denominator to avoid division by
            near-zero values.  Applied as ``max(baseline, epsilon)``.

    Returns:
        dff: (T, H, W) float32 array.  Values are typically in [-1, +∞);
            quiescent pixels hover near 0 and active neurons show positive
            transients of 0.2–2.0+.
    """
    # Denominator: broadcast baseline over time axis, apply epsilon floor
    denom = np.maximum(baseline[np.newaxis], epsilon)  # (1, H, W) → broadcasts

    dff = (stack - baseline[np.newaxis]) / denom
    dff = dff.astype(np.float32)

    logger.info(
        "ΔF/F₀ computed: shape=%s, min=%.3f, max=%.3f, mean=%.4f",
        dff.shape,
        float(dff.min()),
        float(dff.max()),
        float(dff.mean()),
    )
    return dff
