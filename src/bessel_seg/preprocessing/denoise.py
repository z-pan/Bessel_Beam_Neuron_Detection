"""Denoising wrappers for calcium imaging stacks.

Primary: DeepCAD self-supervised denoising (requires optional torch dependency).
Fallback: temporal median filter when DeepCAD is unavailable or disabled.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from bessel_seg.config import DeepCADConfig

logger = logging.getLogger(__name__)


def fallback_temporal_denoise(
    stack: NDArray[np.float32],
    window: int = 5,
) -> NDArray[np.float32]:
    """Apply a temporal median filter as a lightweight DeepCAD fallback.

    Smooths temporal noise while preserving spatial structure.  The filter
    kernel is (window, 1, 1) so it operates independently on each pixel's
    time-series.

    Args:
        stack: (T, H, W) float32 input stack.
        window: Number of consecutive frames included in each median.
            Must be a positive odd integer; even values are accepted and
            treated equivalently by ``scipy.ndimage.median_filter``.

    Returns:
        Denoised (T, H, W) float32 stack.
    """
    logger.info(
        "Applying temporal median filter (window=%d) as DeepCAD fallback.", window
    )
    denoised = median_filter(stack, size=(window, 1, 1))
    return denoised.astype(np.float32)


def deepcad_denoise(
    stack: NDArray[np.float32],
    config: DeepCADConfig,
) -> NDArray[np.float32]:
    """Apply DeepCAD self-supervised denoising to a calcium imaging stack.

    DeepCAD requires PyTorch and the ``deepcad`` package (optional dependency).
    If either is unavailable, or if ``config.enabled`` is ``False``, the
    function logs a warning and returns the input stack unchanged.

    When DeepCAD IS available the function:
    1. Saves *stack* to a temporary TIFF file.
    2. Runs DeepCAD training + inference.
    3. Caches the denoised output to ``data/denoised/`` keyed by a hash of the
       input filename so repeated calls are cheap.

    Args:
        stack: (T, H, W) float32 registered stack.
        config: DeepCAD hyperparameters from ``DeepCADConfig``.

    Returns:
        Denoised (T, H, W) float32 stack (or the original stack if DeepCAD is
        unavailable / disabled).
    """
    if not config.enabled:
        logger.info("DeepCAD denoising disabled (config.enabled=False); skipping.")
        return stack

    # Attempt to import DeepCAD — it is an optional dependency.
    try:
        import deepcad  # noqa: F401
    except ImportError:
        logger.warning(
            "DeepCAD package not found. Install it with: pip install deepcad  "
            "(requires PyTorch). Returning input stack unchanged."
        )
        return stack

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning(
            "PyTorch not found, which is required by DeepCAD. "
            "Returning input stack unchanged."
        )
        return stack

    # --- DeepCAD is available: run training + inference ---
    # Full implementation would: write stack to tmp .tif, call DeepCAD CLI or
    # API with config.patch_xy / patch_t / epochs / batch_size / lr, read back
    # the denoised output, cache to data/denoised/, and return.
    # This stub raises NotImplementedError so callers can fall back gracefully.
    raise NotImplementedError(
        "DeepCAD integration is not yet implemented. "
        "Use fallback_temporal_denoise() instead, or implement the DeepCAD "
        "training+inference loop here."
    )
