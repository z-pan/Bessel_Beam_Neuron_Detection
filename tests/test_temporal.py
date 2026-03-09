"""Unit tests for the temporal sub-package.

Synthetic stack characteristics:
  - Shape: (T=60, H=80, W=100) float32
  - Non-uniform background (bright top, dark bottom strip)
  - Horizontal band artifact superimposed on the temporal mean
  - One active neuron spot with a calcium-transient time-course
  - Seed: 42 throughout for determinism

All tests run without GPU / optional dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from bessel_seg.temporal.background_mask import (
    generate_illumination_mask,
    suppress_band_artifacts,
)
from bessel_seg.temporal.baseline import estimate_baseline
from bessel_seg.temporal.delta_f import compute_delta_f
from bessel_seg.temporal.summary_maps import compute_summary_maps

# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------
T, H, W = 60, 80, 100
SEED = 42

# Bottom rows that are intentionally dark (outside illumination cone)
DARK_ROW_START = 65  # rows [65, 80) are dark


# ---------------------------------------------------------------------------
# Synthetic stack factory
# ---------------------------------------------------------------------------

def _make_synthetic_stack(seed: int = SEED) -> np.ndarray:
    """Return a (T, H, W) float32 stack with:

    - Non-uniform background: brighter toward the top of the frame.
    - Dark bottom strip: rows [DARK_ROW_START, H) are near-zero.
    - Horizontal band artifact: sinusoidal row modulation added to every frame.
    - One neuron at (y=30, x=50) that fires in frames [10, 16) with ΔF/F₀ ≈ 1.0.
    """
    rng = np.random.default_rng(seed)

    # Non-uniform background gradient (dim overall, 8-bit like)
    row_gradient = np.linspace(12.0, 6.0, H).reshape(H, 1)  # brighter at top
    bg = (row_gradient * np.ones((H, W))).astype(np.float32)

    # Replicate across time with additive Gaussian noise
    noise = rng.normal(0.0, 1.5, size=(T, H, W)).astype(np.float32)
    stack = bg[np.newaxis] + noise  # (T, H, W)

    # Dark bottom strip — simulate outside-illumination-cone region
    stack[:, DARK_ROW_START:, :] = rng.uniform(0.0, 0.5, size=(T, H - DARK_ROW_START, W))

    # Horizontal band artifact: add a sinusoidal row modulation
    band_amplitude = 4.0
    row_idx = np.arange(H, dtype=np.float32)
    band_profile = band_amplitude * np.sin(2 * np.pi * row_idx / 10.0)  # ~10-px period
    stack += band_profile[np.newaxis, :, np.newaxis]  # broadcast over T and W

    # Neuron transient at (y=30, x=50): 3×3 region, fires frames 10–15
    neuron_signal = 10.0  # ΔF ~ 10 counts above background (~12 → ΔF/F₀ ≈ 0.8)
    stack[10:17, 28:33, 48:53] += neuron_signal

    # Ensure non-negative
    stack = np.clip(stack, 0.0, None)
    return stack.astype(np.float32)


# ===========================================================================
# baseline
# ===========================================================================

class TestEstimateBaseline:
    def test_output_shape_dtype(self):
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack, percentile=10)
        assert F0.shape == (H, W)
        assert F0.dtype == np.float32

    def test_minimum_clamped_to_1(self):
        """All baseline values must be ≥ 1.0 after clamping."""
        stack = _make_synthetic_stack()
        # Introduce pixels that should go to 0
        stack[:, 0, 0] = 0.0
        F0 = estimate_baseline(stack, percentile=10)
        assert float(F0.min()) >= 1.0

    def test_percentile_0_is_min(self):
        """percentile=0 should equal np.min along axis=0, clipped to ≥1."""
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack, percentile=0)
        expected = np.percentile(stack, 0, axis=0).astype(np.float32)
        expected = np.clip(expected, 1.0, None)
        np.testing.assert_allclose(F0, expected, rtol=1e-5)

    def test_invalid_percentile_raises(self):
        stack = _make_synthetic_stack()
        with pytest.raises(ValueError, match="percentile"):
            estimate_baseline(stack, percentile=101)

    def test_neuron_frames_do_not_bias_baseline(self):
        """The baseline at the neuron location should be close to background,
        not contaminated by the transient (which is in < 50 % of frames)."""
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack, percentile=10)
        # Neuron region baseline should be near background (~9–14)
        neuron_baseline = float(F0[30, 50])
        assert neuron_baseline < 20.0, (
            f"Neuron-pixel baseline {neuron_baseline:.2f} seems too high "
            "(transient is contaminating the baseline estimate)."
        )


# ===========================================================================
# delta_f
# ===========================================================================

class TestComputeDeltaF:
    def test_output_shape_dtype(self):
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack)
        dff = compute_delta_f(stack, F0)
        assert dff.shape == (T, H, W)
        assert dff.dtype == np.float32

    def test_quiescent_pixels_near_zero(self):
        """Pixels that never fire should have dff ≈ 0 on average."""
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack)
        dff = compute_delta_f(stack, F0)
        # Use a region away from the neuron and far from the dark strip
        quiescent_mean = float(np.mean(dff[:, 55:65, 70:90]))
        assert abs(quiescent_mean) < 0.5, (
            f"Quiescent region mean dff = {quiescent_mean:.4f}, expected ≈ 0"
        )

    def test_neuron_region_positive_transient(self):
        """The neuron region in active frames should have positive dff."""
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack)
        dff = compute_delta_f(stack, F0)
        neuron_dff = float(np.mean(dff[10:17, 28:33, 48:53]))
        assert neuron_dff > 0.2, (
            f"Neuron dff during transient = {neuron_dff:.4f}, expected > 0.2"
        )

    def test_epsilon_prevents_zero_division(self):
        """Zero baseline pixels must not produce inf/nan."""
        rng = np.random.default_rng(SEED)
        stack = rng.uniform(0.0, 5.0, size=(T, H, W)).astype(np.float32)
        # Force baseline to zero for some pixels
        F0 = np.zeros((H, W), dtype=np.float32)
        dff = compute_delta_f(stack, F0, epsilon=1.0)
        assert not np.any(np.isinf(dff)), "dff contains inf values"
        assert not np.any(np.isnan(dff)), "dff contains nan values"

    def test_constant_stack_gives_zero_dff(self):
        """If stack == baseline everywhere, dff must be identically zero."""
        F0 = np.full((H, W), 10.0, dtype=np.float32)
        stack = np.stack([F0] * T, axis=0)
        dff = compute_delta_f(stack, F0)
        np.testing.assert_allclose(dff, 0.0, atol=1e-6)


# ===========================================================================
# summary_maps
# ===========================================================================

class TestComputeSummaryMaps:
    def _get_maps(self) -> dict:
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack)
        dff = compute_delta_f(stack, F0)
        return compute_summary_maps(stack, dff)

    def test_keys_present(self):
        maps = self._get_maps()
        assert set(maps.keys()) == {"sigma", "max_proj", "mean", "cv"}

    def test_output_shapes(self):
        maps = self._get_maps()
        for key, arr in maps.items():
            assert arr.shape == (H, W), f"Map '{key}' has wrong shape {arr.shape}"

    def test_output_dtype_float32(self):
        maps = self._get_maps()
        for key, arr in maps.items():
            assert arr.dtype == np.float32, f"Map '{key}' dtype = {arr.dtype}"

    def test_all_maps_in_01_range(self):
        """Every map must be normalised to [0, 1]."""
        maps = self._get_maps()
        for key, arr in maps.items():
            lo, hi = float(arr.min()), float(arr.max())
            assert lo >= -1e-6, f"Map '{key}' min = {lo:.6f} < 0"
            assert hi <= 1.0 + 1e-6, f"Map '{key}' max = {hi:.6f} > 1"

    def test_sigma_map_elevated_at_neuron(self):
        """The sigma map should be higher at the neuron location than far away."""
        stack = _make_synthetic_stack()
        F0 = estimate_baseline(stack)
        dff = compute_delta_f(stack, F0)
        maps = compute_summary_maps(stack, dff)

        neuron_sigma = float(maps["sigma"][30, 50])
        background_sigma = float(np.mean(maps["sigma"][55:65, 70:90]))
        assert neuron_sigma > background_sigma, (
            f"Neuron sigma ({neuron_sigma:.4f}) not > background ({background_sigma:.4f})"
        )

    def test_constant_stack_produces_zero_sigma(self):
        """A perfectly constant stack should yield sigma=0 everywhere."""
        F0 = np.full((H, W), 10.0, dtype=np.float32)
        stack = np.stack([F0] * T, axis=0)
        dff = compute_delta_f(stack, F0)
        maps = compute_summary_maps(stack, dff)
        # sigma and cv should collapse to all-zeros after normalisation
        np.testing.assert_allclose(maps["sigma"], 0.0, atol=1e-6)


# ===========================================================================
# generate_illumination_mask
# ===========================================================================

class TestGenerateIlluminationMask:
    def test_output_shape_dtype(self):
        stack = _make_synthetic_stack()
        mask = generate_illumination_mask(stack, threshold_percentile=5)
        assert mask.shape == (H, W)
        assert mask.dtype == np.bool_

    def test_dark_bottom_excluded(self):
        """The permanently dark bottom rows must be masked out (False)."""
        stack = _make_synthetic_stack()
        mask = generate_illumination_mask(stack, threshold_percentile=5)
        # Bottom strip [DARK_ROW_START, H) should be largely excluded
        dark_region = mask[DARK_ROW_START:, :]
        fraction_excluded = 1.0 - dark_region.mean()
        assert fraction_excluded > 0.80, (
            f"Expected > 80 % of dark bottom strip to be excluded, "
            f"got {fraction_excluded * 100:.1f}% excluded."
        )

    def test_illuminated_top_included(self):
        """The brighter top portion of the frame should mostly be included."""
        stack = _make_synthetic_stack()
        mask = generate_illumination_mask(stack, threshold_percentile=5)
        # Top rows [0, DARK_ROW_START) are illuminated
        bright_region = mask[:DARK_ROW_START, :]
        fraction_included = bright_region.mean()
        assert fraction_included > 0.70, (
            f"Expected > 70 % of illuminated region to be included, "
            f"got {fraction_included * 100:.1f}%."
        )

    def test_not_all_zeros(self):
        """Mask must not be all-False (the redesign goal)."""
        stack = _make_synthetic_stack()
        mask = generate_illumination_mask(stack, threshold_percentile=5)
        assert mask.any(), "Illumination mask is all-False — redesign failed."

    def test_not_all_ones(self):
        """At least the dark strip should be excluded."""
        stack = _make_synthetic_stack()
        mask = generate_illumination_mask(stack, threshold_percentile=5)
        assert not mask.all(), "Illumination mask is all-True — dark region not excluded."

    def test_high_threshold_excludes_more(self):
        """A higher threshold_percentile should exclude at least as many pixels."""
        stack = _make_synthetic_stack()
        mask5 = generate_illumination_mask(stack, threshold_percentile=5)
        mask30 = generate_illumination_mask(stack, threshold_percentile=30)
        assert mask30.sum() <= mask5.sum(), (
            "Higher threshold should exclude at least as many pixels."
        )


# ===========================================================================
# suppress_band_artifacts
# ===========================================================================

class TestSuppressBandArtifacts:
    def _make_banded_image(self) -> np.ndarray:
        """Return a (H, W) image with strong horizontal bands + weak spots."""
        rng = np.random.default_rng(SEED)
        row_idx = np.arange(H, dtype=np.float32)
        # Strong sinusoidal band (amplitude 10)
        band = 10.0 * np.sin(2 * np.pi * row_idx / 10.0)
        image = band[:, np.newaxis] * np.ones((H, W), dtype=np.float32)
        # Add small uniform noise
        image += rng.uniform(0.0, 1.0, size=(H, W)).astype(np.float32)
        # Shift to be non-negative
        image -= image.min()
        return image.astype(np.float32)

    def test_output_shape_dtype(self):
        image = self._make_banded_image()
        result = suppress_band_artifacts(image)
        assert result.shape == (H, W)
        assert result.dtype == np.float32

    def test_no_negative_values(self):
        """Output must be clipped to ≥ 0."""
        image = self._make_banded_image()
        result = suppress_band_artifacts(image)
        assert float(result.min()) >= 0.0, f"Found negative value: {result.min()}"

    def test_row_variance_reduced(self):
        """After correction the variance of row-wise means should drop."""
        image = self._make_banded_image()
        row_means_before = np.mean(image, axis=1)  # (H,)
        result = suppress_band_artifacts(image)
        row_means_after = np.mean(result, axis=1)  # (H,)

        var_before = float(np.var(row_means_before))
        var_after = float(np.var(row_means_after))
        # After subtracting the smooth row profile and clipping to zero,
        # the residual variance from pixel noise prevents perfect elimination.
        # We require at least 50 % reduction — a meaningful signal vs. 10-px bands.
        assert var_after < var_before * 0.5, (
            f"Row-mean variance before={var_before:.4f}, after={var_after:.4f}. "
            "Expected > 50 % reduction."
        )

    def test_invalid_orientation_raises(self):
        image = self._make_banded_image()
        with pytest.raises(ValueError, match="orientation"):
            suppress_band_artifacts(image, orientation="vertical")

    def test_flat_image_unchanged_structure(self):
        """A completely flat image should remain flat (all zeros after subtraction)."""
        image = np.full((H, W), 5.0, dtype=np.float32)
        result = suppress_band_artifacts(image)
        # Row profile = 5.0 everywhere → smooth profile = 5.0 → corrected = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_spot_signal_partially_preserved(self):
        """A small bright spot on a banded background should survive correction."""
        image = self._make_banded_image()
        # Add a clearly bright spot
        image[H // 2, W // 2] += 50.0
        result = suppress_band_artifacts(image)
        # The spot should still be the brightest pixel
        assert int(np.argmax(result)) == np.ravel_multi_index(
            (H // 2, W // 2), (H, W)
        ), "Spot is no longer the maximum after band suppression."
