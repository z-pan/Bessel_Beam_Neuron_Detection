"""Per-pixel baseline fluorescence estimation.

F₀ is computed as a low percentile of the temporal distribution of each
pixel, which avoids contamination from calcium transients that are present
only in a minority of frames (~4–42% across datasets).

The minimum value is clamped to 1.0 to prevent division-by-zero or
near-zero denominators in the downstream ΔF/F₀ computation.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Hard floor for baseline to avoid division by near-zero values in delta_f.py
_BASELINE_FLOOR: float = 1.0


def estimate_baseline(
    stack: NDArray[np.float32],
    percentile: int = 10,
) -> NDArray[np.float32]:
    """Estimate per-pixel baseline fluorescence F₀.

    Uses the *percentile*-th value along the time axis so that transient
    calcium events (present in < 50 % of frames) do not bias the estimate
    upward.

    Args:
        stack: (T, H, W) float32 image stack.  Values should be in the
            original intensity range (e.g. 0–255 for 8-bit data).
        percentile: Percentile (0–100) used to summarise the temporal
            distribution.  Default 10 matches the pipeline config.

    Returns:
        F0: (H, W) float32 baseline image.  All values are ≥ 1.0.

    Raises:
        ValueError: If *percentile* is outside [0, 100].
    """
    if not (0 <= percentile <= 100):
        raise ValueError(
            f"percentile must be in [0, 100], got {percentile}."
        )

    # np.percentile along axis=0 gives one value per (y, x) pixel — (H, W)
    baseline = np.percentile(stack, percentile, axis=0).astype(np.float32)

    n_clamped = int(np.sum(baseline < _BASELINE_FLOOR))
    if n_clamped > 0:
        logger.debug(
            "Clamped %d pixels with baseline < %.1f to %.1f.",
            n_clamped,
            _BASELINE_FLOOR,
            _BASELINE_FLOOR,
        )
    baseline = np.clip(baseline, _BASELINE_FLOOR, None)

    logger.info(
        "Baseline estimated (percentile=%d): min=%.3f, max=%.3f, mean=%.3f",
        percentile,
        float(baseline.min()),
        float(baseline.max()),
        float(baseline.mean()),
    )
    return baseline
