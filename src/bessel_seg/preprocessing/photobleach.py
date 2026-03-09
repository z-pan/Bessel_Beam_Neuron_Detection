"""Photobleaching correction for calcium imaging stacks.

CRITICAL: 9 out of 10 real datasets show 16–37% intensity decline over 300 frames.
Without correction, the ΔF/F₀ baseline is biased and later frames will have
artificially suppressed signals.

This module MUST be applied BEFORE baseline estimation (Module 2).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Minimum number of frames required before fitting a trend
_MIN_FRAMES_FOR_FIT = 10


def _fit_linear(means: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit a linear decay to per-frame means and return the trend.

    Args:
        means: 1-D array of per-frame mean intensities, shape (T,).

    Returns:
        trend: 1-D array of fitted values, same shape as *means*.
    """
    T = len(means)
    t = np.arange(T, dtype=np.float64)
    slope, intercept = np.polyfit(t, means, 1)
    return intercept + slope * t


def _exp_model(t: NDArray, A: float, tau: float, C: float) -> NDArray:
    """Exponential decay model: A * exp(-t / tau) + C."""
    return A * np.exp(-t / tau) + C


def _fit_exponential(means: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit an exponential decay to per-frame means and return the trend.

    Falls back to linear fit if the exponential fit fails.

    Args:
        means: 1-D array of per-frame mean intensities, shape (T,).

    Returns:
        trend: 1-D array of fitted values, same shape as *means*.
    """
    T = len(means)
    t = np.arange(T, dtype=np.float64)

    # Initial parameter guesses: A ≈ range, tau ≈ T/2, C ≈ min
    A0 = float(means.max() - means.min())
    tau0 = float(T / 2)
    C0 = float(means.min())

    try:
        popt, _ = curve_fit(
            _exp_model,
            t,
            means,
            p0=[A0, tau0, C0],
            bounds=([0, 1, 0], [means.max() * 2, T * 10, means.max()]),
            maxfev=5000,
        )
        trend = _exp_model(t, *popt)
        logger.debug(
            "Exponential fit: A=%.4f, tau=%.2f, C=%.4f", popt[0], popt[1], popt[2]
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Exponential fit failed (%s); falling back to linear fit.", exc
        )
        trend = _fit_linear(means)

    return trend


def correct_photobleaching(
    stack: NDArray[np.float32],
    method: str = "linear",
) -> NDArray[np.float32]:
    """Correct photobleaching by normalising the per-frame mean intensity trend.

    The correction divides every frame by a scalar derived from the fitted trend
    so that the per-frame mean is approximately constant throughout the recording.
    Frame 0 is left unchanged (correction_factor[0] == 1.0).

    Args:
        stack: (T, H, W) registered float32 stack.
        method: One of:
            ``"linear"``      — fit a line to per-frame means.
            ``"exponential"`` — fit an exponential decay.
            ``"none"``        — return *stack* unchanged.

    Returns:
        Corrected float32 stack of the same shape as *stack*.

    Raises:
        ValueError: If *method* is not one of the recognised options.
    """
    if method == "none":
        logger.info("Photobleaching correction disabled (method='none').")
        return stack

    if method not in ("linear", "exponential"):
        raise ValueError(
            f"Unknown photobleach correction method '{method}'. "
            "Choose 'linear', 'exponential', or 'none'."
        )

    T = stack.shape[0]
    if T < _MIN_FRAMES_FOR_FIT:
        logger.warning(
            "Too few frames (%d < %d) for reliable trend fitting; "
            "skipping photobleaching correction.",
            T,
            _MIN_FRAMES_FOR_FIT,
        )
        return stack

    # Per-frame mean intensity: shape (T,)
    means = np.mean(stack, axis=(1, 2)).astype(np.float64)

    if method == "linear":
        trend = _fit_linear(means)
    else:  # exponential
        trend = _fit_exponential(means)

    # Log the magnitude of photobleaching detected
    relative_change = (trend[-1] - trend[0]) / (trend[0] + 1e-9)
    logger.info(
        "Photobleaching correction (method='%s'): relative intensity change "
        "over recording = %.1f%%",
        method,
        relative_change * 100,
    )

    # Avoid division by near-zero trend values
    safe_trend = np.where(trend > 1e-6, trend, trend[0] + 1e-6)

    # correction_factor[0] == 1.0; later frames are scaled up to match frame 0
    correction_factor = safe_trend[0] / safe_trend  # shape (T,)

    corrected = stack * correction_factor[:, np.newaxis, np.newaxis].astype(np.float32)
    logger.debug(
        "Post-correction per-frame mean range: [%.4f, %.4f]",
        float(np.mean(corrected, axis=(1, 2)).min()),
        float(np.mean(corrected, axis=(1, 2)).max()),
    )
    return corrected
