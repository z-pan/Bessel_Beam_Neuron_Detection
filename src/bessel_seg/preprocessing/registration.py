"""Rigid (translation-only) registration for calcium imaging stacks.

Uses phase cross-correlation for sub-pixel shift estimation and
scipy.ndimage.shift (bilinear interpolation) for applying shifts.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation

logger = logging.getLogger(__name__)

# Warn if any frame requires a shift larger than this many pixels
_LARGE_SHIFT_THRESHOLD = 5.0


def _build_reference(
    stack: NDArray[np.float32],
    reference: str,
) -> NDArray[np.float32]:
    """Build the reference frame used for shift estimation.

    Args:
        stack: (T, H, W) float32 array.
        reference: ``"mean"`` — temporal mean; ``"best_snr"`` — frame with
            highest SNR (approximated as mean/std of the frame).

    Returns:
        2-D (H, W) reference image.
    """
    if reference == "mean":
        return np.mean(stack, axis=0)

    if reference == "best_snr":
        frame_means = np.mean(stack, axis=(1, 2))
        frame_stds = np.std(stack, axis=(1, 2)) + 1e-9
        snr_per_frame = frame_means / frame_stds
        best_idx = int(np.argmax(snr_per_frame))
        logger.debug("Best-SNR reference frame: index %d (SNR=%.3f)", best_idx, snr_per_frame[best_idx])
        return stack[best_idx]

    raise ValueError(
        f"Unknown reference mode '{reference}'. Choose 'mean' or 'best_snr'."
    )


def rigid_register(
    stack: NDArray[np.float32],
    method: str = "phase_correlation",
    reference: str = "mean",
    upsample_factor: int = 10,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Apply rigid (translation-only) registration to a time-series stack.

    Args:
        stack: (T, H, W) input image stack, float32.
        method: ``"phase_correlation"`` — estimate shifts via
            ``skimage.registration.phase_cross_correlation``; ``"none"`` —
            return input unchanged with zero shifts.
        reference: ``"mean"`` — use temporal mean as reference;
            ``"best_snr"`` — use the frame with the highest SNR.
        upsample_factor: Sub-pixel precision multiplier passed to
            ``phase_cross_correlation``.

    Returns:
        registered_stack: (T, H, W) float32 aligned stack.
        shifts: (T, 2) float64 array of (dy, dx) shifts applied to each frame.
            Positive dy/dx means the frame was shifted in the +row / +column
            direction to align with the reference.

    Raises:
        ValueError: If *method* or *reference* is not a recognised option.
    """
    T = stack.shape[0]
    shifts = np.zeros((T, 2), dtype=np.float64)

    if method == "none":
        logger.info("Registration disabled (method='none').")
        return stack, shifts

    if method != "phase_correlation":
        raise ValueError(
            f"Unknown registration method '{method}'. "
            "Choose 'phase_correlation' or 'none'."
        )

    ref = _build_reference(stack, reference)
    registered = np.empty_like(stack)

    for t in range(T):
        frame = stack[t]
        # phase_cross_correlation returns (shift, error, phasediff)
        # shift is in (row, col) = (dy, dx) convention
        detected_shift, _, _ = phase_cross_correlation(
            ref,
            frame,
            upsample_factor=upsample_factor,
        )
        dy, dx = float(detected_shift[0]), float(detected_shift[1])
        shifts[t] = [dy, dx]

        # Apply shift with bilinear interpolation (order=1)
        registered[t] = ndimage_shift(frame, shift=(dy, dx), order=1, mode="nearest")

    max_shift = float(np.max(np.abs(shifts)))
    logger.info(
        "Registration complete. Max shift magnitude: %.2f px "
        "(over %d frames, upsample_factor=%d).",
        max_shift,
        T,
        upsample_factor,
    )
    if max_shift > _LARGE_SHIFT_THRESHOLD:
        logger.warning(
            "Large shifts detected (max %.2f px > %.1f px threshold). "
            "Consider checking for sample drift or mis-ordered frames.",
            max_shift,
            _LARGE_SHIFT_THRESHOLD,
        )

    return registered, shifts
