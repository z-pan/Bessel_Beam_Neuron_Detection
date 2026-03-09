"""ROI overlay rendering for Bessel beam neuron segmentation results."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from bessel_seg.data_types import NeuronROI

# Default colour map: detection_type → (R, G, B) uint8
_DEFAULT_COLOURS: dict[str, tuple[int, int, int]] = {
    "paired": (70, 130, 255),          # blue
    "single_near_pair": (255, 165, 0),  # orange
    "single_isolated": (50, 205, 50),   # green
}
_FALLBACK_COLOUR: tuple[int, int, int] = (200, 200, 200)  # grey


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_to_uint8(frame: NDArray) -> NDArray[np.uint8]:
    """Stretch a 2-D image to [0, 255] uint8."""
    f = frame.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-6:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def _draw_circle(
    canvas: NDArray[np.uint8],   # (H, W, 3)
    cy: float,
    cx: float,
    radius: float,
    colour: tuple[int, int, int],
) -> None:
    """Draw a circle outline on *canvas* using the midpoint algorithm (in-place)."""
    H, W = canvas.shape[:2]
    r = int(round(radius))
    if r < 1:
        return
    cy_i, cx_i = int(round(cy)), int(round(cx))

    def _put(y: int, x: int) -> None:
        if 0 <= y < H and 0 <= x < W:
            canvas[y, x] = colour

    # Midpoint circle algorithm — 8-way symmetry
    x, y = r, 0
    p = 1 - r
    while x >= y:
        _put(cy_i + y, cx_i + x)
        _put(cy_i - y, cx_i + x)
        _put(cy_i + y, cx_i - x)
        _put(cy_i - y, cx_i - x)
        _put(cy_i + x, cx_i + y)
        _put(cy_i - x, cx_i + y)
        _put(cy_i + x, cx_i - y)
        _put(cy_i - x, cx_i - y)
        y += 1
        if p <= 0:
            p += 2 * y + 1
        else:
            x -= 1
            p += 2 * y - 2 * x + 1


def _draw_cross(
    canvas: NDArray[np.uint8],
    cy: float,
    cx: float,
    size: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw a small cross-hair marker (in-place)."""
    H, W = canvas.shape[:2]
    cy_i, cx_i = int(round(cy)), int(round(cx))
    for dx in range(-size, size + 1):
        x = cx_i + dx
        if 0 <= cy_i < H and 0 <= x < W:
            canvas[cy_i, x] = colour
    for dy in range(-size, size + 1):
        y = cy_i + dy
        if 0 <= y < H and 0 <= cx_i < W:
            canvas[y, cx_i] = colour


def _draw_text_label(
    canvas: NDArray[np.uint8],
    text: str,
    y: int,
    x: int,
    colour: tuple[int, int, int],
) -> None:
    """
    Render a minimal pixel-font text label (3×5 digits/letters only) at (y, x).
    Falls back to a single bright pixel if the character is unknown.
    """
    # 3-wide × 5-tall bitmaps for digits 0-9 and a dot
    _FONT: dict[str, list[str]] = {
        "0": ["###", "# #", "# #", "# #", "###"],
        "1": [" # ", "## ", " # ", " # ", "###"],
        "2": ["###", "  #", "###", "#  ", "###"],
        "3": ["###", "  #", "###", "  #", "###"],
        "4": ["# #", "# #", "###", "  #", "  #"],
        "5": ["###", "#  ", "###", "  #", "###"],
        "6": ["###", "#  ", "###", "# #", "###"],
        "7": ["###", "  #", " # ", " # ", " # "],
        "8": ["###", "# #", "###", "# #", "###"],
        "9": ["###", "# #", "###", "  #", "###"],
        ".": ["   ", "   ", "   ", "   ", " # "],
        " ": ["   ", "   ", "   ", "   ", "   "],
    }
    H, W = canvas.shape[:2]
    col = x
    for ch in str(text):
        bitmap = _FONT.get(ch, _FONT[" "])
        for row_i, row_str in enumerate(bitmap):
            for col_i, pixel in enumerate(row_str):
                if pixel == "#":
                    py, px = y + row_i, col + col_i
                    if 0 <= py < H and 0 <= px < W:
                        canvas[py, px] = colour
        col += 4  # 3 px wide + 1 px gap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_roi_overlay(
    frame: NDArray,
    neurons: list[NeuronROI],
    colormap: Optional[dict[str, tuple[int, int, int]]] = None,
    show_labels: bool = True,
    show_masks: bool = True,
    mask_alpha: float = 0.25,
) -> NDArray[np.uint8]:
    """Render an RGB overlay of neuron ROIs on a single image frame.

    Args:
        frame: (H, W) grayscale image (any numeric dtype).
        neurons: List of NeuronROI objects from the pipeline.
        colormap: Optional dict mapping detection_type str to (R, G, B) uint8 tuple.
                  Defaults to blue=paired, orange=single_near_pair, green=single_isolated.
        show_labels: If True, draw neuron_id number near each ROI centre.
        show_masks: If True, blend ROI masks as a semi-transparent fill.
        mask_alpha: Transparency of the mask fill (0=transparent, 1=opaque).

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    colours = {**_DEFAULT_COLOURS, **(colormap or {})}

    # Build greyscale base as RGB
    grey = _normalise_to_uint8(frame)
    canvas = np.stack([grey, grey, grey], axis=-1)  # (H, W, 3)
    H, W = canvas.shape[:2]

    for roi in neurons:
        colour = colours.get(roi.detection_type, _FALLBACK_COLOUR)

        # --- Mask fill (semi-transparent blend) ---
        if show_masks:
            combined_mask = np.zeros((H, W), dtype=bool)
            if roi.mask_left is not None:
                combined_mask |= roi.mask_left
            if roi.mask_right is not None:
                combined_mask |= roi.mask_right
            if combined_mask.any():
                overlay_layer = np.zeros((H, W, 3), dtype=np.float32)
                overlay_layer[combined_mask] = colour
                canvas = (
                    canvas.astype(np.float32) * (1.0 - mask_alpha)
                    + overlay_layer * mask_alpha
                ).clip(0, 255).astype(np.uint8)

        # --- Circle outlines at spot positions ---
        if roi.left_spot is not None:
            _draw_circle(canvas, roi.left_spot.y, roi.left_spot.x,
                         max(roi.left_radius, 3.0), colour)
        if roi.right_spot is not None:
            _draw_circle(canvas, roi.right_spot.y, roi.right_spot.x,
                         max(roi.right_radius, 3.0), colour)

        # --- Cross-hair at neuron centre ---
        _draw_cross(canvas, roi.center_y, roi.center_x, 3, colour)

        # --- Text label: neuron_id ---
        if show_labels:
            lx = int(round(roi.center_x)) + 5
            ly = int(round(roi.center_y)) - 3
            _draw_text_label(canvas, str(roi.neuron_id), ly, lx, colour)

    return canvas


def make_roi_label_image(
    neurons: list[NeuronROI],
    H: int,
    W: int,
) -> NDArray[np.int32]:
    """Build a (H, W) integer label image: 0=background, i+1=neuron i.

    Args:
        neurons: Ordered list of NeuronROI objects.
        H, W: Image dimensions.

    Returns:
        (H, W) int32 label image.
    """
    label_img = np.zeros((H, W), dtype=np.int32)
    for idx, roi in enumerate(neurons):
        label = idx + 1
        if roi.mask_left is not None:
            label_img[roi.mask_left] = label
        if roi.mask_right is not None:
            label_img[roi.mask_right] = label
    return label_img
