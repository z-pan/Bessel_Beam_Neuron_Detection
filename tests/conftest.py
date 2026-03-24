"""Shared fixtures and helpers for the Bessel beam neuron detection test suite.

Provides:
* Synthetic data generators that mimic real Bessel beam two-photon images
* Convenience factories for Spot, SpotPair, NeuronROI objects
* Reusable pytest fixtures used across multiple test modules

All random operations use a fixed seed (42) for determinism.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bessel_seg.config import PipelineConfig
from bessel_seg.data_types import FrameQuality, NeuronROI, Spot, SpotPair

# ---------------------------------------------------------------------------
# Constants matching real data characteristics (from CLAUDE.md)
# ---------------------------------------------------------------------------
SIGMA_SPOT: float = 2.5       # Median measured spot sigma
PAIR_DX: float = 43.0         # Median pair spacing (px)
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Object factories
# ---------------------------------------------------------------------------

def make_spot(
    y: float = 50.0,
    x: float = 80.0,
    sigma: float = SIGMA_SPOT,
    intensity: float = 1.0,
    frame_idx: int | None = None,
    snr: float = 0.0,
) -> Spot:
    """Create a Spot with sensible defaults."""
    return Spot(
        y=y, x=x, sigma=sigma, intensity=intensity,
        frame_idx=frame_idx, snr=snr,
    )


def make_pair(
    cy: float = 50.0,
    cx: float = 80.0,
    dx: float = PAIR_DX,
    frame_idx: int | None = None,
    pair_cost: float = 0.1,
    intensity_ratio: float = 0.9,
    scale_similarity: float = 0.95,
) -> SpotPair:
    """Create a SpotPair centred at (cy, cx) with horizontal spacing dx."""
    left = make_spot(cy, cx - dx / 2, frame_idx=frame_idx)
    right = make_spot(cy, cx + dx / 2, frame_idx=frame_idx)
    return SpotPair(
        left=left,
        right=right,
        pair_distance=dx,
        horizontal_distance=dx,
        vertical_offset=0.0,
        intensity_ratio=intensity_ratio,
        scale_similarity=scale_similarity,
        pair_cost=pair_cost,
    )


# ---------------------------------------------------------------------------
# Synthetic image / stack generators
# ---------------------------------------------------------------------------

def gaussian_footprint(
    H: int,
    W: int,
    cy: float,
    cx: float,
    sigma: float,
) -> NDArray[np.float32]:
    """2-D Gaussian footprint normalised to peak=1."""
    Y, X = np.ogrid[:H, :W]
    return np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)
    ).astype(np.float32)


def make_gaussian_image(
    H: int = 100,
    W: int = 200,
    cy: float = 50.0,
    cx: float = 100.0,
    sigma: float = SIGMA_SPOT,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> NDArray[np.float32]:
    """Generate a single-frame 2-D Gaussian image."""
    return (offset + amplitude * gaussian_footprint(H, W, cy, cx, sigma)).astype(
        np.float32
    )


def add_events(
    stack: NDArray[np.float32],
    cy: float,
    cx: float,
    sigma: float,
    amplitude: float,
    starts: list[int],
    duration: int,
) -> None:
    """Inject calcium-transient events into *stack* in-place.

    Each event has a triangular rise-peak-decay profile over *duration* frames.
    """
    T, H, W = stack.shape
    fp = gaussian_footprint(H, W, cy, cx, sigma)
    for start in starts:
        for dt in range(duration):
            t = start + dt
            if t >= T:
                break
            half = max(duration / 2.0, 1.0)
            weight = min(dt + 1, duration - dt) / half
            stack[t] += amplitude * weight * fp


def make_blank_stack(
    T: int = 20,
    H: int = 60,
    W: int = 120,
    fill: float = 0.0,
) -> NDArray[np.float32]:
    """Return a constant-valued (T, H, W) stack."""
    return np.full((T, H, W), fill, dtype=np.float32)


def make_dff_with_transient(
    T: int = 30,
    H: int = 60,
    W: int = 120,
    cy: int = 30,
    cx: int = 60,
    start: int = 10,
    amps: list[float] | None = None,
    radius: int = 4,
) -> NDArray[np.float32]:
    """Build a ΔF/F₀ stack with a single calcium transient.

    The transient is a disc of radius *radius* centred at *(cy, cx)*
    with amplitude profile given by *amps* starting at frame *start*.
    """
    if amps is None:
        amps = [0.3, 0.6, 1.0, 0.6, 0.3]
    dff = np.zeros((T, H, W), dtype=np.float32)
    Y, X = np.ogrid[:H, :W]
    mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
    for i, a in enumerate(amps):
        t = start + i
        if 0 <= t < T:
            dff[t][mask] = a
    return dff


def make_synthetic_pipeline_stack(seed: int = DEFAULT_SEED) -> NDArray[np.float32]:
    """Generate a (60, 100, 160) stack for full pipeline integration tests.

    Contains:
    - 2 paired neurons (A at (50, 80), B at (30, 110))
    - 2 single-spot neurons (C at (75, 40), D at (20, 40))
    - 1 persistent background blob at (60, 148) — should be rejected
    - Linear photobleaching (-25%)
    - Non-uniform background gradient + horizontal band artifacts
    - Gaussian noise (std=0.3)
    """
    np.random.seed(seed)
    T, H, W = 60, 100, 160

    # Background: gradient + horizontal bands
    bg = np.ones((H, W), dtype=np.float32) * 12.0
    bg += np.linspace(3.0, 0.0, W, dtype=np.float32)[np.newaxis, :]
    for row in range(0, H, 18):
        bg[row : row + 2, :] += 4.0

    # Photobleaching
    bleach = np.linspace(1.0, 0.75, T, dtype=np.float32)[:, np.newaxis, np.newaxis]
    stack = np.tile(bg[np.newaxis], (T, 1, 1)) * bleach
    stack += np.random.normal(0.0, 0.3, (T, H, W)).astype(np.float32)
    stack = np.clip(stack, 0.0, None).astype(np.float32)

    # Neuron A (paired): center (50, 80)
    add_events(stack, 50.0, 80.0 - PAIR_DX / 2, SIGMA_SPOT, 30.0, [5, 25, 45], 5)
    add_events(stack, 50.0, 80.0 + PAIR_DX / 2, SIGMA_SPOT, 30.0, [5, 25, 45], 5)

    # Neuron B (paired): center (30, 110)
    add_events(stack, 30.0, 110.0 - PAIR_DX / 2, SIGMA_SPOT, 30.0, [10, 30, 50], 5)
    add_events(stack, 30.0, 110.0 + PAIR_DX / 2, SIGMA_SPOT, 30.0, [10, 30, 50], 5)

    # Neuron C (single, isolated): center (75, 40)
    add_events(stack, 75.0, 40.0, SIGMA_SPOT, 28.0, [15, 35, 53], 5)

    # Neuron D (single, isolated): center (20, 40)
    add_events(stack, 20.0, 40.0, SIGMA_SPOT, 28.0, [8, 28, 48], 5)

    # Persistent blob at (60, 148) — always bright -> should be rejected
    stack += 20.0 * gaussian_footprint(H, W, 60.0, 148.0, 3.0)[np.newaxis, :, :]

    return stack.astype(np.float32)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_pipeline_stack() -> NDArray[np.float32]:
    """Session-scoped full pipeline test stack (60, 100, 160)."""
    return make_synthetic_pipeline_stack(seed=DEFAULT_SEED)


@pytest.fixture
def small_stack() -> NDArray[np.float32]:
    """A small (50, 100, 120) stack with values in [5, 30] for unit tests."""
    rng = np.random.default_rng(DEFAULT_SEED)
    return rng.uniform(5.0, 30.0, size=(50, 100, 120)).astype(np.float32)


@pytest.fixture
def test_config() -> PipelineConfig:
    """Pipeline config tuned for synthetic test data."""
    cfg = PipelineConfig()
    cfg.preprocessing.registration_method = "none"
    cfg.preprocessing.photobleach_correction = "linear"
    cfg.deepcad.enabled = False
    cfg.detection.threshold = 0.004
    cfg.detection.local_snr_threshold = 3.0
    cfg.detection.min_sigma = 1.0
    cfg.detection.max_sigma = 5.0
    cfg.detection.num_sigma = 6
    cfg.pairing.horizontal_dist_min = 25
    cfg.pairing.horizontal_dist_max = 60
    cfg.pairing.vertical_offset_max = 8
    cfg.clustering.eps = 10.0
    cfg.clustering.min_samples = 2
    cfg.temporal_validation.min_peak_dff = 0.15
    cfg.temporal_validation.max_active_fraction = 0.60
    cfg.temporal_validation.min_rise_frames = 1
    cfg.temporal_validation.min_decay_frames = 1
    cfg.refinement.min_confidence = 0.0
    cfg.refinement.top_k_frames = 6
    cfg.refinement.gaussian_fit_radius = 8
    return cfg
