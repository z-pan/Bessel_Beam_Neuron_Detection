"""Unit tests for the preprocessing sub-package.

Synthetic stack: (50, 100, 120) float32, mimicking real 8-bit data.
All random operations use np.random.seed(42) for determinism.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from bessel_seg.config import DeepCADConfig
from bessel_seg.preprocessing.channel_extract import (
    _natural_sort_key,
    extract_green_channel,
)
from bessel_seg.preprocessing.denoise import fallback_temporal_denoise
from bessel_seg.preprocessing.photobleach import correct_photobleaching
from bessel_seg.preprocessing.registration import rigid_register

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
T, H, W = 50, 100, 120
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stack(seed: int = RNG_SEED) -> np.ndarray:
    """Return a (T, H, W) float32 stack with values in [5, 30]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(5.0, 30.0, size=(T, H, W)).astype(np.float32)


# ===========================================================================
# channel_extract
# ===========================================================================

class TestNaturalSortKey:
    def test_numeric_ordering(self):
        names = ["frame_10.tif", "frame_2.tif", "frame_1.tif"]
        assert sorted(names, key=_natural_sort_key) == [
            "frame_1.tif",
            "frame_2.tif",
            "frame_10.tif",
        ]

    def test_no_digits(self):
        names = ["c.tif", "a.tif", "b.tif"]
        assert sorted(names, key=_natural_sort_key) == ["a.tif", "b.tif", "c.tif"]


class TestExtractGreenChannel:
    def test_grayscale_multiframe_tif(self, tmp_path: Path):
        """Multi-frame greyscale TIFF → (T, H, W) float32."""
        stack = _make_stack()
        tif_path = tmp_path / "stack.tif"
        tifffile.imwrite(str(tif_path), stack.astype(np.uint8))

        result = extract_green_channel(tif_path)
        assert result.shape == (T, H, W)
        assert result.dtype == np.float32

    def test_rgb_channel_last_multiframe_tif(self, tmp_path: Path):
        """(T, H, W, 3) RGB TIFF → green channel extracted."""
        rng = np.random.default_rng(RNG_SEED)
        rgb = rng.integers(0, 255, size=(T, H, W, 3), dtype=np.uint8)
        # Set green channel to a distinct value
        rgb[:, :, :, 1] = 200
        tif_path = tmp_path / "rgb_stack.tif"
        tifffile.imwrite(str(tif_path), rgb)

        result = extract_green_channel(tif_path)
        assert result.shape == (T, H, W)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 200.0)

    def test_folder_of_single_frame_tifs(self, tmp_path: Path):
        """Folder of single-frame greyscale TIFs → correctly stacked in order."""
        rng = np.random.default_rng(RNG_SEED)
        # Create 15 frames so we exceed the ≥10 assertion
        n_frames = 15
        frames = rng.integers(0, 255, size=(n_frames, H, W), dtype=np.uint8)
        for i in range(n_frames):
            # Write with zero-padded names to test natural sort
            fname = tmp_path / f"frame_{i:03d}.tif"
            tifffile.imwrite(str(fname), frames[i])

        result = extract_green_channel(tmp_path)
        assert result.shape == (n_frames, H, W)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0], frames[0].astype(np.float32))
        np.testing.assert_allclose(result[-1], frames[-1].astype(np.float32))

    def test_folder_natural_sort_order(self, tmp_path: Path):
        """Files named frame_1 … frame_15 are loaded in numeric order."""
        rng = np.random.default_rng(RNG_SEED)
        n_frames = 15
        frames = rng.integers(0, 255, size=(n_frames, H, W), dtype=np.uint8)
        # Write WITHOUT zero-padding to exercise natural sort
        for i in range(n_frames):
            fname = tmp_path / f"frame_{i + 1}.tif"
            tifffile.imwrite(str(fname), frames[i])

        result = extract_green_channel(tmp_path)
        assert result.shape == (n_frames, H, W)
        # frame_1 should be first
        np.testing.assert_allclose(result[0], frames[0].astype(np.float32))

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_green_channel("/nonexistent/path/stack.tif")

    def test_too_few_frames_raises(self, tmp_path: Path):
        """Only 5 frames in a TIFF should trigger the ≥10 assertion."""
        rng = np.random.default_rng(RNG_SEED)
        stack = rng.integers(0, 255, size=(5, H, W), dtype=np.uint8)
        tif_path = tmp_path / "tiny.tif"
        tifffile.imwrite(str(tif_path), stack)
        with pytest.raises(AssertionError):
            extract_green_channel(tif_path)


# ===========================================================================
# photobleach
# ===========================================================================

class TestCorrectPhotobleaching:
    def test_none_returns_unchanged(self):
        stack = _make_stack()
        result = correct_photobleaching(stack, method="none")
        np.testing.assert_array_equal(result, stack)

    def test_linear_reduces_mean_variation(self):
        """After linear correction the per-frame mean CV should drop significantly."""
        rng = np.random.default_rng(RNG_SEED)
        base = rng.uniform(15.0, 25.0, size=(T, H, W)).astype(np.float32)

        # Inject -30% linear decay
        t = np.arange(T, dtype=np.float32)
        decay = 1.0 - 0.30 * (t / (T - 1))  # 1.0 → 0.70
        stack = base * decay[:, np.newaxis, np.newaxis]

        result = correct_photobleaching(stack, method="linear")

        # Per-frame mean should be nearly constant (< 2% CV)
        post_means = np.mean(result, axis=(1, 2))
        cv = post_means.std() / post_means.mean()
        assert cv < 0.02, f"Post-correction CV of per-frame means = {cv:.4f}, expected < 0.02"

    def test_linear_frame_mean_change_below_2pct(self):
        """Verify the specific <2% criterion from the task spec."""
        rng = np.random.default_rng(RNG_SEED)
        base = rng.uniform(10.0, 20.0, size=(T, H, W)).astype(np.float32)

        t = np.arange(T, dtype=np.float32)
        decay = 1.0 - 0.30 * (t / (T - 1))
        stack = base * decay[:, np.newaxis, np.newaxis]

        result = correct_photobleaching(stack, method="linear")
        post_means = np.mean(result, axis=(1, 2))

        # Peak-to-peak variation relative to mean
        ptp_relative = (post_means.max() - post_means.min()) / post_means.mean()
        assert ptp_relative < 0.02, (
            f"Post-correction peak-to-peak variation = {ptp_relative:.4f}, expected < 0.02"
        )

    def test_exponential_reduces_mean_variation(self):
        """Exponential correction on an exponentially decaying stack."""
        rng = np.random.default_rng(RNG_SEED)
        base = rng.uniform(15.0, 25.0, size=(T, H, W)).astype(np.float32)

        t = np.arange(T, dtype=np.float32)
        # Exponential decay: drop ~28% by final frame
        decay = 0.7 + 0.3 * np.exp(-t / (T / 3))
        stack = base * decay[:, np.newaxis, np.newaxis]

        result = correct_photobleaching(stack, method="exponential")
        post_means = np.mean(result, axis=(1, 2))
        cv = post_means.std() / post_means.mean()
        assert cv < 0.05, f"Post-correction CV = {cv:.4f}, expected < 0.05"

    def test_frame_zero_unchanged(self):
        """correction_factor[0] == 1.0, so frame 0 must be unchanged."""
        rng = np.random.default_rng(RNG_SEED)
        base = rng.uniform(10.0, 20.0, size=(T, H, W)).astype(np.float32)
        t = np.arange(T, dtype=np.float32)
        decay = 1.0 - 0.30 * (t / (T - 1))
        stack = base * decay[:, np.newaxis, np.newaxis]

        result = correct_photobleaching(stack, method="linear")
        np.testing.assert_allclose(result[0], stack[0], rtol=1e-5)

    def test_invalid_method_raises(self):
        stack = _make_stack()
        with pytest.raises(ValueError, match="Unknown photobleach"):
            correct_photobleaching(stack, method="polynomial")

    def test_output_dtype_float32(self):
        stack = _make_stack()
        result = correct_photobleaching(stack, method="linear")
        assert result.dtype == np.float32

    def test_output_shape_preserved(self):
        stack = _make_stack()
        result = correct_photobleaching(stack, method="linear")
        assert result.shape == stack.shape


# ===========================================================================
# registration
# ===========================================================================

class TestRigidRegister:
    def test_none_returns_unchanged(self):
        stack = _make_stack()
        reg, shifts = rigid_register(stack, method="none")
        np.testing.assert_array_equal(reg, stack)
        np.testing.assert_array_equal(shifts, np.zeros((T, 2)))

    def test_output_shapes(self):
        stack = _make_stack()
        reg, shifts = rigid_register(stack, method="phase_correlation", upsample_factor=1)
        assert reg.shape == stack.shape
        assert shifts.shape == (T, 2)
        assert reg.dtype == np.float32
        assert shifts.dtype == np.float64

    def test_known_shift_recovery(self):
        """Inject a (dy=3, dx=-2) shift in one frame and verify recovery.

        We create a stack where every frame is identical and then shift one
        frame by a known amount.  After registration the residual shift for
        that frame should be < 0.5 px.
        """
        rng = np.random.default_rng(RNG_SEED)
        # Use a stack with some spatial structure so phase correlation works
        base = np.zeros((H, W), dtype=np.float32)
        # Add a few bright spots
        for _ in range(10):
            cy = rng.integers(10, H - 10)
            cx = rng.integers(10, W - 10)
            base[cy - 3 : cy + 4, cx - 3 : cx + 4] = rng.uniform(50, 150)

        stack = np.stack([base] * T, axis=0)

        # Shift frame 5 by (dy=3, dx=-2)
        from scipy.ndimage import shift as ndimage_shift

        dy_true, dx_true = 3.0, -2.0
        stack[5] = ndimage_shift(base, shift=(dy_true, dx_true), order=1, mode="nearest")

        registered, shifts = rigid_register(
            stack, method="phase_correlation", upsample_factor=10
        )

        # phase_cross_correlation returns the *correction* shift, i.e. the
        # negative of the injected offset, so shifts[5] ≈ (-dy_true, -dx_true).
        # Verify the magnitude of the recovered correction is within 0.5 px.
        residual_dy = abs(shifts[5, 0] + dy_true)   # shifts[5,0] ≈ -dy_true
        residual_dx = abs(shifts[5, 1] + dx_true)   # shifts[5,1] ≈ -dx_true
        assert residual_dy < 0.5, f"dy residual = {residual_dy:.4f} px"
        assert residual_dx < 0.5, f"dx residual = {residual_dx:.4f} px"

    def test_zero_shift_for_identical_frames(self):
        """All-identical frames should produce near-zero shifts."""
        rng = np.random.default_rng(RNG_SEED)
        frame = rng.uniform(5.0, 30.0, size=(H, W)).astype(np.float32)
        stack = np.stack([frame] * T, axis=0)

        _, shifts = rigid_register(stack, method="phase_correlation", upsample_factor=1)
        max_shift = np.max(np.abs(shifts))
        assert max_shift < 1.0, f"Expected near-zero shifts, got max={max_shift:.4f}"

    def test_invalid_method_raises(self):
        stack = _make_stack()
        with pytest.raises(ValueError, match="Unknown registration method"):
            rigid_register(stack, method="optical_flow")

    def test_best_snr_reference(self):
        """reference='best_snr' should run without error and return correct shapes."""
        stack = _make_stack()
        reg, shifts = rigid_register(
            stack,
            method="phase_correlation",
            reference="best_snr",
            upsample_factor=1,
        )
        assert reg.shape == stack.shape
        assert shifts.shape == (T, 2)


# ===========================================================================
# denoise
# ===========================================================================

class TestFallbackTemporalDenoise:
    def test_output_shape_dtype(self):
        stack = _make_stack()
        result = fallback_temporal_denoise(stack, window=5)
        assert result.shape == stack.shape
        assert result.dtype == np.float32

    def test_smooths_temporal_noise(self):
        """After median filtering a noisy stack, temporal std per-pixel should decrease."""
        rng = np.random.default_rng(RNG_SEED)
        # Constant signal + large temporal noise
        signal = np.full((T, H, W), 20.0, dtype=np.float32)
        noise = rng.normal(0.0, 10.0, size=(T, H, W)).astype(np.float32)
        stack = signal + noise

        result = fallback_temporal_denoise(stack, window=5)
        # Temporal std should be reduced
        std_before = np.std(stack, axis=0).mean()
        std_after = np.std(result, axis=0).mean()
        assert std_after < std_before, (
            f"Expected temporal noise reduction; std before={std_before:.3f}, after={std_after:.3f}"
        )

    def test_window_1_is_identity(self):
        """Window=1 median filter should leave the stack unchanged."""
        stack = _make_stack()
        result = fallback_temporal_denoise(stack, window=1)
        np.testing.assert_array_almost_equal(result, stack)


class TestDeepCadDenoise:
    def test_disabled_returns_unchanged(self):
        """config.enabled=False must return input stack unchanged."""
        from bessel_seg.preprocessing.denoise import deepcad_denoise

        stack = _make_stack()
        cfg = DeepCADConfig(enabled=False)
        result = deepcad_denoise(stack, cfg)
        np.testing.assert_array_equal(result, stack)

    def test_missing_deepcad_returns_unchanged(self):
        """When deepcad is not installed, the function should return input unchanged."""
        import sys
        from unittest.mock import patch

        from bessel_seg.preprocessing.denoise import deepcad_denoise

        stack = _make_stack()
        cfg = DeepCADConfig(enabled=True)

        # Simulate deepcad not being installed
        with patch.dict(sys.modules, {"deepcad": None}):
            result = deepcad_denoise(stack, cfg)
        np.testing.assert_array_equal(result, stack)
