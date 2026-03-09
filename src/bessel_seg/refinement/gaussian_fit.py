"""2D Gaussian fitting for Bessel beam spot localisation.

Each confirmed neuron is represented by one or two approximately Gaussian
fluorescent spots.  After coarse localisation via LoG detection and pairing,
this module refines the centre, width (sigma), and amplitude of each spot by
fitting the 2D Gaussian model::

    f(y, x) = offset + amplitude · exp(−[(x−x₀)² + (y−y₀)²] / (2·σ²))

to a small image patch centred on the initial estimate.

Fitting uses :func:`scipy.optimize.curve_fit` (Levenberg–Marquardt) with
physically-motivated bounds.  On failure the function returns the original
estimate with a fallback sigma of 2.5 px (the median measured from 3 346
real-data Gaussian fits, CLAUDE.md).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Fallback sigma returned when curve_fit fails (measured median, CLAUDE.md)
_FALLBACK_SIGMA: float = 2.5


def _gaussian_2d(
    YX: tuple[NDArray, NDArray],
    y0: float,
    x0: float,
    sigma: float,
    amplitude: float,
    offset: float,
) -> NDArray:
    """Flat (ravelled) 2-D Gaussian for use with curve_fit.

    Args:
        YX: Tuple of (Y_coords, X_coords) grids, each ravelled.
        y0, x0: Centre row and column.
        sigma: Isotropic Gaussian width (px).
        amplitude: Peak height above offset.
        offset: Constant background level.

    Returns:
        Ravelled predicted values, shape (N,).
    """
    Y, X = YX
    return (offset + amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2)))


def fit_spot_gaussian(
    image: NDArray[np.float32],
    spot_center: tuple[float, float],
    radius: int = 10,
) -> tuple[float, float, float, float]:
    """Fit a 2-D isotropic Gaussian to a spot in *image*.

    A square patch of half-size *radius* centred on *spot_center* is
    extracted from *image*.  The model::

        f(y, x) = offset + amplitude · exp(−[(x−x₀)² + (y−y₀)²] / (2σ²))

    is fitted with :func:`scipy.optimize.curve_fit` using bounds that
    keep the solution physically meaningful.

    On any fitting failure (``RuntimeError``, ``ValueError``, singular
    covariance matrix) the function logs a debug message and returns the
    original centre estimate with ``sigma = _FALLBACK_SIGMA`` and
    ``amplitude = image[iy, ix]`` (the raw pixel value at the initial
    centre).

    Args:
        image: 2-D float32 image (single frame or temporal average).
        spot_center: ``(cy, cx)`` initial centre estimate (sub-pixel).
        radius: Half-size of the fitting window (px).  The patch spans
            ``[cy−radius, cy+radius] × [cx−radius, cx+radius]``,
            clipped to the image boundary.

    Returns:
        ``(y0, x0, sigma, amplitude)`` — refined sub-pixel centre, Gaussian
        width, and peak amplitude above background.  All in image
        coordinates.
    """
    cy, cx = spot_center
    H, W = image.shape

    iy = int(round(cy))
    ix = int(round(cx))

    y_lo = max(0, iy - radius)
    y_hi = min(H, iy + radius + 1)
    x_lo = max(0, ix - radius)
    x_hi = min(W, ix + radius + 1)

    patch = image[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

    if patch.size < 4:
        logger.debug("fit_spot_gaussian: patch too small at (%.1f, %.1f) — returning fallback", cy, cx)
        amp_fallback = float(np.clip(image[iy, ix], 0.0, None)) if 0 <= iy < H and 0 <= ix < W else 0.0
        return cy, cx, _FALLBACK_SIGMA, amp_fallback

    # Coordinate grids in global image coordinates
    Y, X = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    Y = Y.astype(np.float64)
    X = X.astype(np.float64)

    # Initial parameter guess
    patch_min = float(patch.min())
    patch_max = float(patch.max())
    sigma_init = max(float(radius) / 4.0, _FALLBACK_SIGMA)
    amp_init = max(patch_max - patch_min, 1e-3)
    p0 = [cy, cx, sigma_init, amp_init, patch_min]

    # Parameter bounds: y0, x0, sigma, amplitude, offset
    lower = [float(y_lo), float(x_lo), 0.5, 0.0, -abs(patch_max)]
    upper = [float(y_hi), float(x_hi), float(radius), 2.0 * max(patch_max, 1e-6), patch_max]

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            (Y.ravel(), X.ravel()),
            patch.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=2000,
        )
        y0_fit, x0_fit, sigma_fit, amp_fit, _ = popt

        logger.debug(
            "fit_spot_gaussian: (%.1f, %.1f) → (%.2f, %.2f) σ=%.2f amp=%.3f",
            cy, cx, y0_fit, x0_fit, sigma_fit, amp_fit,
        )
        return float(y0_fit), float(x0_fit), float(sigma_fit), float(amp_fit)

    except (RuntimeError, ValueError) as exc:
        logger.debug(
            "fit_spot_gaussian: curve_fit failed at (%.1f, %.1f): %s — using fallback",
            cy, cx, exc,
        )
        amp_fallback = float(np.clip(patch_max, 0.0, None))
        return cy, cx, _FALLBACK_SIGMA, amp_fallback
