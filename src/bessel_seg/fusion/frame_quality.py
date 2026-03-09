"""Per-frame quality scoring for the Bessel beam neuron detection pipeline.

Scores each frame on three independent metrics:
  - SNR: mean intensity in spot regions vs. background std
  - Sharpness: Laplacian variance (sensitive to focus quality)
  - PairRate: fraction of detected spots that were successfully paired

The overall score is a weighted combination used by downstream modules to
select the best frames for Gaussian fitting and cluster refinement.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import laplace

from bessel_seg.data_types import FrameQuality, Spot, SpotPair

logger = logging.getLogger(__name__)

# Radius (px) around each spot centre used to define the "signal region"
_SIGNAL_RADIUS: int = 4
# Overall score weights (must sum to 1.0)
_W_SNR: float = 0.4
_W_SHARPNESS: float = 0.3
_W_PAIR_RATE: float = 0.3


def _extract_spot_mask(
    shape: tuple[int, int],
    spots: list[Spot],
    radius: int = _SIGNAL_RADIUS,
) -> NDArray[np.bool_]:
    """Return a boolean mask that is True inside circles around each spot.

    Args:
        shape: (H, W) of the frame.
        spots: Spot objects whose centres define the circles.
        radius: Pixel radius of each circle.

    Returns:
        (H, W) boolean array.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    ys, xs = np.ogrid[:H, :W]
    for s in spots:
        iy, ix = int(round(s.y)), int(round(s.x))
        dist2 = (ys - iy) ** 2 + (xs - ix) ** 2
        mask |= dist2 <= radius ** 2
    return mask


def _frame_snr(
    frame: NDArray[np.float32],
    spots: list[Spot],
) -> float:
    """Compute signal-to-noise ratio for one frame.

    Signal region: union of circles (radius ``_SIGNAL_RADIUS``) around spots.
    Background region: all pixels outside the signal region.

    Args:
        frame: (H, W) single frame.
        spots: Detected spots in this frame.

    Returns:
        SNR value ≥ 0.  Returns 0.0 when no spots are given or background
        has zero standard deviation.
    """
    if not spots:
        return 0.0

    signal_mask = _extract_spot_mask(frame.shape, spots)
    background_mask = ~signal_mask

    signal_vals = frame[signal_mask]
    bg_vals = frame[background_mask]

    if signal_vals.size == 0 or bg_vals.size == 0:
        return 0.0

    bg_std = float(np.std(bg_vals))
    if bg_std < 1e-9:
        return 0.0

    snr = float(np.mean(signal_vals) - np.mean(bg_vals)) / bg_std
    return max(snr, 0.0)


def _frame_sharpness(frame: NDArray[np.float32]) -> float:
    """Return the Laplacian variance of a frame (focus/sharpness proxy).

    Args:
        frame: (H, W) single frame.

    Returns:
        Non-negative float; higher = sharper.
    """
    lap = laplace(frame.astype(np.float64))
    return float(np.var(lap))


def compute_frame_quality(
    stack: NDArray[np.float32],
    spots_per_frame: dict[int, list[Spot]],
    pairs_per_frame: dict[int, list[SpotPair]],
) -> list[FrameQuality]:
    """Compute quality metrics for every frame in *stack*.

    Three metrics per frame:

    * **SNR** — mean(signal pixels) − mean(background) divided by std(background),
      where signal pixels are circles of radius ``_SIGNAL_RADIUS`` around each
      detected spot.
    * **Sharpness** — variance of the Laplacian response (focus proxy).
    * **PairRate** — ``len(pairs) / max(len(spots), 1)`` for the frame.

    The overall score is::

        overall = 0.4 · norm(SNR) + 0.3 · norm(Sharpness) + 0.3 · PairRate

    where SNR and Sharpness are normalised to [0, 1] across all frames before
    combining.

    Args:
        stack: (T, H, W) float32 image stack.
        spots_per_frame: Dict ``frame_idx → list[Spot]`` (may be sparse).
        pairs_per_frame: Dict ``frame_idx → list[SpotPair]`` (may be sparse).

    Returns:
        List of :class:`~bessel_seg.data_types.FrameQuality` objects, one per
        frame, in ascending ``frame_idx`` order.
    """
    T = stack.shape[0]
    if T == 0:
        return []

    raw_snr: list[float] = []
    raw_sharpness: list[float] = []
    raw_pair_rate: list[float] = []

    for t in range(T):
        frame = stack[t]
        spots = spots_per_frame.get(t, [])
        pairs = pairs_per_frame.get(t, [])

        snr = _frame_snr(frame, spots)
        sharpness = _frame_sharpness(frame)
        pair_rate = len(pairs) / max(len(spots), 1)

        raw_snr.append(snr)
        raw_sharpness.append(sharpness)
        raw_pair_rate.append(pair_rate)

    # Normalise SNR and Sharpness to [0, 1] across all frames
    snr_arr = np.array(raw_snr, dtype=np.float64)
    shp_arr = np.array(raw_sharpness, dtype=np.float64)

    def _norm(arr: NDArray) -> NDArray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    norm_snr = _norm(snr_arr)
    norm_sharpness = _norm(shp_arr)

    qualities: list[FrameQuality] = []
    for t in range(T):
        overall = (
            _W_SNR * float(norm_snr[t])
            + _W_SHARPNESS * float(norm_sharpness[t])
            + _W_PAIR_RATE * raw_pair_rate[t]
        )
        qualities.append(
            FrameQuality(
                frame_idx=t,
                snr=raw_snr[t],
                sharpness=raw_sharpness[t],
                pair_rate=raw_pair_rate[t],
                overall_score=float(np.clip(overall, 0.0, 1.0)),
            )
        )

    logger.info(
        "compute_frame_quality: %d frames — mean SNR=%.3f, mean sharpness=%.3f, "
        "mean pair_rate=%.3f, mean overall=%.3f",
        T,
        float(snr_arr.mean()),
        float(shp_arr.mean()),
        float(np.mean(raw_pair_rate)),
        float(np.mean([q.overall_score for q in qualities])),
    )
    return qualities
