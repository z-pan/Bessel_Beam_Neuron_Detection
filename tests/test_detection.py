"""Unit tests for the detection sub-package.

Synthetic image: (100, 200) float32 with three Gaussian spots at known
positions, plus low-level background noise and a horizontal band artifact.

All random operations use seed 42 for determinism.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from bessel_seg.config import DetectionConfig, EDIConfig
from bessel_seg.data_types import Spot
from bessel_seg.detection.adaptive_threshold import filter_spots_adaptive
from bessel_seg.detection.blob_detect import detect_spots, detect_spots_per_frame
from bessel_seg.detection.enhanced_image import build_enhanced_detection_image

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
H, W = 100, 200
SEED = 42

# True spot positions: (y, x) pairs — all safely away from borders
TRUE_SPOTS: list[tuple[int, int]] = [(30, 50), (50, 120), (70, 160)]
SPOT_SIGMA = 3.0
# Large amplitude on near-zero background ensures spots dominate after
# normalisation (background ≈ 0, spot peaks ≈ 1).  With bg=0 and noise
# std=0.3, a Gaussian peak of ~8 units yields SNR >> threshold=0.4.
SPOT_AMPLITUDE = 500.0
BG_LEVEL = 0.0
NOISE_STD = 0.3


# ---------------------------------------------------------------------------
# Synthetic image factory
# ---------------------------------------------------------------------------

def _make_image(
    spots: list[tuple[int, int]] = TRUE_SPOTS,
    sigma: float = SPOT_SIGMA,
    amplitude: float = SPOT_AMPLITUDE,
    bg_level: float = BG_LEVEL,
    noise_std: float = NOISE_STD,
    add_bands: bool = False,
    seed: int = SEED,
) -> np.ndarray:
    """Return a (H, W) float32 image with Gaussian spots on noisy background.

    Default parameters use dark background (bg_level=0) and high amplitude so
    that after normalisation spots are near 1.0 and background is near 0.
    This gives clear SNR for adaptive filtering and band-suppression tests.
    """
    rng = np.random.default_rng(seed)
    image = np.full((H, W), bg_level, dtype=np.float32)
    image += rng.normal(0.0, noise_std, size=(H, W)).astype(np.float32)

    # Add Gaussian spots
    for cy, cx in spots:
        bump = np.zeros((H, W), dtype=np.float32)
        bump[cy, cx] = amplitude
        image += gaussian_filter(bump, sigma=sigma)

    if add_bands:
        # Sinusoidal horizontal bands (amplitude 3, period 15)
        row_idx = np.arange(H, dtype=np.float32)
        band = 3.0 * np.sin(2 * np.pi * row_idx / 15.0)
        image += band[:, np.newaxis]

    image = np.clip(image, 0.0, None)
    return image.astype(np.float32)


def _normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def _default_det_config(**overrides) -> DetectionConfig:
    """DetectionConfig tuned to detect SPOT_SIGMA=3 spots reliably."""
    cfg = DetectionConfig(
        min_sigma=1.0,
        max_sigma=6.0,
        num_sigma=10,
        threshold=0.02,
        overlap=0.5,
        local_snr_threshold=0.4,
    )
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _closest_detected(detected: list[Spot], true_y: float, true_x: float) -> float:
    """Return minimum Euclidean distance from (true_y, true_x) to any detected spot."""
    if not detected:
        return float("inf")
    dists = [((s.y - true_y) ** 2 + (s.x - true_x) ** 2) ** 0.5 for s in detected]
    return min(dists)


# ===========================================================================
# build_enhanced_detection_image
# ===========================================================================

class TestBuildEDI:
    def _make_summary_maps(self, seed: int = SEED) -> dict[str, np.ndarray]:
        """Create plausible (H, W) normalised summary maps."""
        img = _normalise(_make_image())
        rng = np.random.default_rng(seed)
        return {
            "sigma":    img,
            "max_proj": _normalise(img + rng.uniform(0, 0.1, img.shape).astype(np.float32)),
            "cv":       _normalise(rng.uniform(0, 1, img.shape).astype(np.float32)),
            "mean":     _normalise(img * 0.5),
        }

    def test_output_shape_dtype(self):
        maps = self._make_summary_maps()
        mask = np.ones((H, W), dtype=np.bool_)
        cfg = EDIConfig()
        edi = build_enhanced_detection_image(maps, mask, cfg)
        assert edi.shape == (H, W)
        assert edi.dtype == np.float32

    def test_output_in_01(self):
        maps = self._make_summary_maps()
        mask = np.ones((H, W), dtype=np.bool_)
        edi = build_enhanced_detection_image(maps, mask, EDIConfig())
        assert float(edi.min()) >= -1e-6
        assert float(edi.max()) <= 1.0 + 1e-6

    def test_masked_pixels_are_zero(self):
        """Pixels outside the illumination mask must be 0 in the EDI."""
        maps = self._make_summary_maps()
        mask = np.zeros((H, W), dtype=np.bool_)
        mask[:50, :] = True          # only top half is illuminated
        edi = build_enhanced_detection_image(maps, mask, EDIConfig())
        # Bottom half must be all-zero
        np.testing.assert_array_equal(edi[50:, :], 0.0)

    def test_all_mask_false_returns_zeros(self):
        """All-False mask → EDI should be all zeros."""
        maps = self._make_summary_maps()
        mask = np.zeros((H, W), dtype=np.bool_)
        edi = build_enhanced_detection_image(maps, mask, EDIConfig())
        np.testing.assert_array_equal(edi, 0.0)

    def test_missing_map_raises(self):
        maps = {"sigma": np.zeros((H, W), dtype=np.float32)}
        mask = np.ones((H, W), dtype=np.bool_)
        with pytest.raises(KeyError, match="max_proj"):
            build_enhanced_detection_image(maps, mask, EDIConfig())

    def test_spots_survive_edi(self):
        """Spot regions should remain locally bright in the EDI.

        After band artifact suppression the exact peak pixel may shift by a
        pixel or two, so we check the max within a 5-px neighbourhood rather
        than the exact centre coordinate.  The neighbourhood max must exceed
        the 90th-percentile of the EDI (top-10% threshold) to confirm the
        spot survived.
        """
        img = _normalise(_make_image())   # dark bg → spots dominate at ~1.0
        maps = {"sigma": img, "max_proj": img, "cv": img, "mean": img}
        mask = np.ones((H, W), dtype=np.bool_)
        edi = build_enhanced_detection_image(maps, mask, EDIConfig())
        p90 = float(np.percentile(edi, 90))
        r = 5  # neighbourhood radius (pixels)
        for cy, cx in TRUE_SPOTS:
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            local_max = float(edi[y0:y1, x0:x1].max())
            assert local_max > p90, (
                f"Spot ({cy},{cx}): local_max={local_max:.4f} not above p90={p90:.4f}"
            )


# ===========================================================================
# detect_spots
# ===========================================================================

class TestDetectSpots:
    def test_detects_all_three_spots(self):
        """All 3 synthetic spots must be detected within 3 px."""
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)

        for cy, cx in TRUE_SPOTS:
            dist = _closest_detected(spots, cy, cx)
            assert dist < 3.0, (
                f"True spot ({cy},{cx}) not detected within 3 px "
                f"(closest = {dist:.2f} px). Found {len(spots)} spots total."
            )

    def test_returns_spot_objects(self):
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        assert all(isinstance(s, Spot) for s in spots)

    def test_frame_idx_is_none(self):
        """detect_spots does not set frame_idx — that's the caller's job."""
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        assert all(s.frame_idx is None for s in spots)

    def test_sigma_in_expected_range(self):
        """Detected sigma should be close to the true SPOT_SIGMA=3."""
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        for cy, cx in TRUE_SPOTS:
            matched = [s for s in spots
                       if ((s.y - cy) ** 2 + (s.x - cx) ** 2) ** 0.5 < 3.0]
            if matched:
                sigma = matched[0].sigma
                assert 1.0 <= sigma <= 7.0, (
                    f"Spot sigma {sigma:.2f} outside [1, 7] for true spot ({cy},{cx})"
                )

    def test_empty_image_returns_empty(self):
        image = np.zeros((H, W), dtype=np.float32)
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        assert spots == []

    def test_high_threshold_reduces_detections(self):
        """A much higher LoG threshold should find fewer spots."""
        image = _normalise(_make_image())
        cfg_low  = _default_det_config(threshold=0.01)
        cfg_high = _default_det_config(threshold=0.20)
        spots_low  = detect_spots(image, cfg_low)
        spots_high = detect_spots(image, cfg_high)
        assert len(spots_high) <= len(spots_low), (
            "Higher threshold should not produce more detections."
        )


# ===========================================================================
# detect_spots_per_frame
# ===========================================================================

class TestDetectSpotsPerFrame:
    def _make_dff_stack(self, n_active: int = 5, seed: int = SEED) -> np.ndarray:
        """(T=20, H, W) dff stack; first n_active frames have spot signal."""
        T = 20
        stack = np.zeros((T, H, W), dtype=np.float32)
        img = _normalise(_make_image())
        for t in range(n_active):
            stack[t] = img
        return stack

    def test_output_type(self):
        stack = self._make_dff_stack()
        mask = np.ones((H, W), dtype=np.bool_)
        cfg = _default_det_config()
        result = detect_spots_per_frame(stack, mask, cfg)
        assert isinstance(result, dict)

    def test_frame_idx_set_correctly(self):
        """Spots in frame t must have frame_idx == t."""
        stack = self._make_dff_stack(n_active=3)
        mask = np.ones((H, W), dtype=np.bool_)
        cfg = _default_det_config()
        result = detect_spots_per_frame(stack, mask, cfg)
        for t, spots in result.items():
            for s in spots:
                assert s.frame_idx == t, (
                    f"Spot in frame {t} has frame_idx={s.frame_idx}"
                )

    def test_low_signal_frames_skipped(self):
        """All-zero frames must be absent from the result dict."""
        stack = self._make_dff_stack(n_active=3)  # frames 3-19 are zero
        mask = np.ones((H, W), dtype=np.bool_)
        cfg = _default_det_config()
        result = detect_spots_per_frame(stack, mask, cfg, quality_threshold=0.1)
        for t in result:
            assert t < 3, f"Zero frame {t} should have been skipped."

    def test_mask_zeroes_out_region(self):
        """Spots in masked-out region should not appear in results."""
        stack = self._make_dff_stack(n_active=5)
        # Mask out entire image → all spots suppressed
        mask = np.zeros((H, W), dtype=np.bool_)
        cfg = _default_det_config()
        result = detect_spots_per_frame(stack, mask, cfg)
        total = sum(len(v) for v in result.values())
        assert total == 0, f"Expected 0 spots with all-zero mask, got {total}"


# ===========================================================================
# filter_spots_adaptive
# ===========================================================================

class TestFilterSpotsAdaptive:
    def test_real_spots_retained(self):
        """All 3 true spots should survive adaptive filtering."""
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        filtered = filter_spots_adaptive(spots, image, cfg)

        for cy, cx in TRUE_SPOTS:
            dist = _closest_detected(filtered, cy, cx)
            assert dist < 3.0, (
                f"True spot ({cy},{cx}) lost after adaptive filter "
                f"(closest remaining: {dist:.2f} px)."
            )

    def test_noise_spot_rejected(self):
        """A spot injected into flat-noise background should be filtered out."""
        rng = np.random.default_rng(SEED)
        # Image: pure low-level noise — no real spots
        noise_img = rng.uniform(0.0, 0.05, size=(H, W)).astype(np.float32)
        # Manually create a faint spot at a specific location
        fake_spot = Spot(y=50.0, x=100.0, sigma=3.0, intensity=0.05)
        cfg = _default_det_config(local_snr_threshold=0.4)
        filtered = filter_spots_adaptive([fake_spot], noise_img, cfg)
        # The spot should be rejected since SNR is too low on a flat image
        assert len(filtered) == 0, (
            f"Noise spot should be rejected; got {len(filtered)} spot(s) retained."
        )

    def test_border_spots_rejected(self):
        """Spots within max_sigma pixels of the border are discarded."""
        image = _normalise(_make_image())
        cfg = _default_det_config(max_sigma=5.0)
        border_spot = Spot(y=2.0, x=2.0, sigma=3.0, intensity=0.9)
        result = filter_spots_adaptive([border_spot], image, cfg)
        assert len(result) == 0, "Border spot should be rejected."

    def test_snr_field_populated(self):
        """Accepted spots must have snr > 0 after filtering."""
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        filtered = filter_spots_adaptive(spots, image, cfg)
        for s in filtered:
            assert s.snr > 0.0, f"Accepted spot has snr={s.snr}"

    def test_empty_input_returns_empty(self):
        image = _normalise(_make_image())
        cfg = _default_det_config()
        assert filter_spots_adaptive([], image, cfg) == []

    def test_filter_reduces_count(self):
        """Filtering should not increase the number of spots."""
        image = _normalise(_make_image(noise_std=3.0))   # noisier → more false blobs
        cfg = _default_det_config(threshold=0.005)        # very low threshold → many candidates
        spots = detect_spots(image, cfg)
        filtered = filter_spots_adaptive(spots, image, cfg)
        assert len(filtered) <= len(spots)

    def test_output_elements_are_spots(self):
        image = _normalise(_make_image())
        cfg = _default_det_config()
        spots = detect_spots(image, cfg)
        filtered = filter_spots_adaptive(spots, image, cfg)
        assert all(isinstance(s, Spot) for s in filtered)
