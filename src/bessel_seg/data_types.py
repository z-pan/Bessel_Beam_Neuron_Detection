"""Shared data structures for the Bessel beam neuron segmentation pipeline.

All inter-module data exchange uses these dataclasses.
Never pass raw dicts between modules.

Coordinate convention: (y, x) — image convention, row-major.
Array shapes are documented inline as comments, e.g. # (T, H, W).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Spot:
    """A single detected fluorescent spot."""

    y: float                          # Row coordinate (pixels, sub-pixel)
    x: float                          # Column coordinate (pixels, sub-pixel)
    sigma: float                      # Gaussian sigma from LoG detection
    intensity: float                  # Peak intensity in EDI or source image
    frame_idx: Optional[int] = None   # Which frame it was detected in (None if from EDI)
    snr: float = 0.0                  # Local signal-to-noise ratio


@dataclass
class SpotPair:
    """A matched pair of spots forming one neuron candidate."""

    left: Spot                        # Left spot (smaller x)
    right: Spot                       # Right spot (larger x)
    pair_distance: float              # Euclidean distance between spots
    horizontal_distance: float        # |right.x - left.x|
    vertical_offset: float            # |right.y - left.y|
    intensity_ratio: float            # min(I_l, I_r) / max(I_l, I_r), in [0, 1]
    scale_similarity: float           # 1 - |σ_l - σ_r| / max(σ_l, σ_r), in [0, 1]
    pair_cost: float = 0.0           # Cost from Hungarian matching

    @property
    def center(self) -> tuple[float, float]:
        """Neuron center = midpoint of the pair.

        Returns:
            (y, x) center coordinates.
        """
        return (
            (self.left.y + self.right.y) / 2,
            (self.left.x + self.right.x) / 2,
        )


@dataclass
class NeuronROI:
    """Final output: a confirmed neuron with full metadata."""

    neuron_id: int
    center_y: float
    center_x: float
    left_spot: Optional[Spot]         # None for single-spot neurons
    right_spot: Optional[Spot]        # None for single-spot neurons
    left_radius: float               # Fitted radius from Gaussian fit (0 if absent)
    right_radius: float
    confidence: float                 # Overall confidence in [0, 1]
    detection_type: str               # "paired" | "single_near_pair" | "single_isolated"
    detection_count: int              # Number of frames where detected
    total_frames: int                 # Total frames in the sequence
    temporal_trace: Optional[NDArray[np.float32]] = None  # shape: (T,)
    mask_left: Optional[NDArray[np.bool_]] = None         # 2D binary mask for left spot
    mask_right: Optional[NDArray[np.bool_]] = None        # 2D binary mask for right spot


@dataclass
class FrameQuality:
    """Quality metrics for a single frame."""

    frame_idx: int
    snr: float                        # Signal-to-noise ratio
    sharpness: float                  # Laplacian variance
    pair_rate: float                  # Fraction of spots successfully paired
    overall_score: float              # Weighted combination


@dataclass
class PipelineResult:
    """Complete output of the pipeline for one dataset."""

    neurons: list[NeuronROI]
    frame_qualities: list[FrameQuality]
    background_mask: NDArray[np.bool_]       # (H, W) persistent background
    summary_maps: dict[str, NDArray]         # 'sigma', 'max_proj', 'cv', 'edi'
    metadata: dict                           # Processing parameters, timing, etc.
