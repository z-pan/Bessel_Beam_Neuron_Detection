"""Green-channel extraction from TIFF files.

Supports two input modes:
  (a) Folder of single-frame .tif files → natural-sorted, stacked into (T, H, W)
  (b) Single multi-frame .tif file → loaded directly as (T, H, W)

Single-frame format handling:
  - (H, W, 3)  — RGB, channel-last
  - (3, H, W)  — RGB, channel-first
  - (H, W)     — already greyscale

All outputs are float32 with values in the original 8-bit intensity range [0, 255].
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Green channel index (0=R, 1=G, 2=B)
_GREEN_IDX = 1


def _natural_sort_key(name: str) -> list[int | str]:
    """Key function for natural (human) sort order.

    Splits the filename into alternating text/number segments so that
    "frame_9.tif" < "frame_10.tif".
    """
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _extract_green_single_frame(frame: NDArray) -> NDArray[np.float32]:
    """Extract the green channel from a single 2-D or 3-D frame array.

    Args:
        frame: Array of shape (H, W), (H, W, C), or (C, H, W).

    Returns:
        2-D float32 array of shape (H, W).
    """
    arr = np.asarray(frame, dtype=np.float32)

    if arr.ndim == 2:
        # Already greyscale
        logger.warning(
            "Frame is already greyscale (shape %s); returning as-is.", arr.shape
        )
        return arr

    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            # (H, W, C) — channel-last
            return arr[:, :, _GREEN_IDX]
        if arr.shape[0] in (3, 4):
            # (C, H, W) — channel-first
            return arr[_GREEN_IDX, :, :]
        # Ambiguous: treat as greyscale volume and warn
        logger.warning(
            "Unexpected 3-D frame shape %s; cannot determine channel axis. "
            "Returning slice [0] as greyscale.",
            arr.shape,
        )
        return arr[0]

    raise ValueError(
        f"Cannot extract green channel from array with {arr.ndim} dimensions "
        f"(shape {arr.shape})."
    )


def _load_folder(folder: Path) -> NDArray[np.float32]:
    """Load a folder of single-frame .tif files into a (T, H, W) stack.

    Args:
        folder: Directory containing .tif / .tiff files.

    Returns:
        (T, H, W) float32 array.

    Raises:
        FileNotFoundError: If no .tif files are found.
        ValueError: If fewer than 10 frames are found.
    """
    tif_files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: _natural_sort_key(p.name),
    )
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in folder: {folder}")

    frames: list[NDArray[np.float32]] = []
    for path in tif_files:
        raw = tifffile.imread(str(path))
        frames.append(_extract_green_single_frame(raw))

    stack = np.stack(frames, axis=0)  # (T, H, W)
    logger.info(
        "Loaded %d frames from folder '%s'. Stack shape: %s, dtype: %s, "
        "intensity range: [%.1f, %.1f]",
        len(frames),
        folder,
        stack.shape,
        stack.dtype,
        float(stack.min()),
        float(stack.max()),
    )
    assert stack.shape[0] >= 10, (
        f"Too few frames: expected ≥ 10, got {stack.shape[0]}. "
        "Check that the folder contains the full dataset."
    )
    return stack


def _load_tif_file(tif_path: Path) -> NDArray[np.float32]:
    """Load a multi-frame .tif file into a (T, H, W) stack.

    Args:
        tif_path: Path to the .tif stack.

    Returns:
        (T, H, W) float32 array.

    Raises:
        ValueError: If fewer than 10 frames are found.
    """
    raw = tifffile.imread(str(tif_path))
    arr = np.asarray(raw, dtype=np.float32)

    # Possible shapes from tifffile for a multi-frame RGB file:
    #   (T, H, W, 3), (T, H, W), (T, 3, H, W)
    if arr.ndim == 2:
        # Single greyscale frame — wrap into (1, H, W)
        logger.warning(
            "TIF file '%s' loaded as a single 2-D frame; wrapping to (1, H, W).",
            tif_path,
        )
        stack = arr[np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            # Single RGB frame (H, W, C) — extract green and wrap
            logger.warning(
                "TIF file '%s' appears to be a single RGB frame (shape %s); "
                "extracting green and wrapping to (1, H, W).",
                tif_path,
                arr.shape,
            )
            stack = _extract_green_single_frame(arr)[np.newaxis]
        else:
            # Multi-frame greyscale (T, H, W)
            stack = arr
    elif arr.ndim == 4:
        if arr.shape[3] in (3, 4):
            # (T, H, W, C) — extract green channel
            stack = arr[:, :, :, _GREEN_IDX]
        elif arr.shape[1] in (3, 4):
            # (T, C, H, W) — extract green channel
            stack = arr[:, _GREEN_IDX, :, :]
        else:
            raise ValueError(
                f"Cannot determine channel axis for 4-D array with shape {arr.shape} "
                f"from '{tif_path}'."
            )
    else:
        raise ValueError(
            f"Unexpected array dimensions {arr.ndim} (shape {arr.shape}) "
            f"from '{tif_path}'."
        )

    logger.info(
        "Loaded TIF file '%s'. Stack shape: %s, dtype: %s, "
        "intensity range: [%.1f, %.1f]",
        tif_path,
        stack.shape,
        stack.dtype,
        float(stack.min()),
        float(stack.max()),
    )
    assert stack.shape[0] >= 10, (
        f"Too few frames: expected ≥ 10, got {stack.shape[0]}. "
        "Check that the .tif file contains the full dataset."
    )
    return stack


def extract_green_channel(tif_path: str | Path) -> NDArray[np.float32]:
    """Load data and extract the green channel into a (T, H, W) float32 stack.

    Supports two input modes:

    (a) **Folder of single-frame .tif files** — files are sorted by natural order
        (so "frame_9.tif" comes before "frame_10.tif") and each frame's green
        channel is extracted and stacked.

    (b) **Single multi-frame .tif file** — loaded with tifffile and the green
        channel extracted from the resulting array.

    Args:
        tif_path: Path to a .tif file *or* a directory containing .tif files.

    Returns:
        Array of shape (T, H, W), dtype float32, values in original intensity
        range (0–255 for 8-bit data).

    Raises:
        FileNotFoundError: If the path does not exist or no .tif files are found.
        ValueError: If fewer than 10 frames are present.
    """
    path = Path(tif_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_dir():
        return _load_folder(path)
    else:
        return _load_tif_file(path)
