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
    import hashlib
    import tempfile
    from pathlib import Path

    import tifffile

    T, H, W = stack.shape

    # Check for cached result keyed by content hash
    content_hash = hashlib.md5(stack.tobytes()[:4096]).hexdigest()[:12]
    cache_dir = Path("data/denoised")
    cache_path = cache_dir / f"deepcad_{content_hash}_{T}_{H}_{W}.tif"

    if cache_path.exists():
        logger.info("Loading cached DeepCAD result from %s", cache_path)
        cached = tifffile.imread(str(cache_path)).astype(np.float32)
        if cached.shape == stack.shape:
            return cached
        logger.warning("Cached shape %s != input shape %s; re-running.", cached.shape, stack.shape)

    try:
        from deepcad.train_collection import training_class
        from deepcad.test_collection import testing_class
    except ImportError:
        logger.warning(
            "DeepCAD API (train_collection/test_collection) not found. "
            "The installed deepcad package may be incompatible. "
            "Falling back to temporal median filter."
        )
        return fallback_temporal_denoise(stack)

    # Write stack to a temporary directory as DeepCAD expects file-based I/O
    with tempfile.TemporaryDirectory(prefix="deepcad_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input"
        input_path.mkdir()
        input_tif = input_path / "stack.tif"
        tifffile.imwrite(str(input_tif), stack)

        output_path = tmpdir_path / "output"
        output_path.mkdir()
        model_path = tmpdir_path / "model"
        model_path.mkdir()

        try:
            # DeepCAD training phase (self-supervised, no ground truth needed)
            train_params = training_class()
            train_params.datasets_folder = str(input_path)
            train_params.n_epochs = config.epochs
            train_params.batch_size = config.batch_size
            train_params.lr = config.lr
            train_params.patch_xy = min(config.patch_xy, min(H, W))
            train_params.patch_t = min(config.patch_t, T)
            train_params.pth_dir = str(model_path)
            train_params.GPU = "0"

            logger.info(
                "DeepCAD training: epochs=%d, batch=%d, lr=%.1e, patch_xy=%d, patch_t=%d",
                config.epochs, config.batch_size, config.lr,
                train_params.patch_xy, train_params.patch_t,
            )
            train_params.run()

            # DeepCAD inference phase
            test_params = testing_class()
            test_params.datasets_folder = str(input_path)
            test_params.pth_dir = str(model_path)
            test_params.denoise_model = str(
                sorted(model_path.glob("*.pth"))[-1]
            ) if list(model_path.glob("*.pth")) else ""
            test_params.output_dir = str(output_path)
            test_params.GPU = "0"
            test_params.patch_xy = train_params.patch_xy
            test_params.patch_t = train_params.patch_t

            logger.info("DeepCAD inference: writing to %s", output_path)
            test_params.run()

            # Load denoised result
            output_files = sorted(output_path.rglob("*.tif"))
            if not output_files:
                logger.warning("DeepCAD produced no output files; returning input unchanged.")
                return stack

            denoised = tifffile.imread(str(output_files[0])).astype(np.float32)

            # Handle shape mismatches from DeepCAD padding
            if denoised.shape != stack.shape:
                logger.warning(
                    "DeepCAD output shape %s differs from input %s; cropping to match.",
                    denoised.shape, stack.shape,
                )
                denoised = denoised[:T, :H, :W]

            # Cache for future runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(cache_path), denoised)
            logger.info("Cached DeepCAD result to %s", cache_path)

            return denoised

        except Exception as exc:
            logger.warning(
                "DeepCAD failed (%s: %s). Falling back to temporal median filter.",
                type(exc).__name__, exc,
            )
            return fallback_temporal_denoise(stack)
