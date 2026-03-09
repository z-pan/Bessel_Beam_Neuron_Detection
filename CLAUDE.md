# CLAUDE.md — Bessel Beam Two-Photon Neuron Segmentation Pipeline

## Project Overview

This project implements an automated neuron segmentation pipeline for **Bessel beam two-photon fluorescence microscopy** images of mouse cortical neurons. Each neuron appears as a **horizontally-arranged pair of circular spots** (due to Bessel beam side-lobes), spaced ~30–50 pixels apart (mean 40.0 px, confirmed across 10 datasets and 5 resolutions). The pipeline processes time-series of single-frame `.tif` files and outputs identified neuron ROIs with confidence scores.

**Core algorithm philosophy**: Temporal-First → Geometric Pairing → Multi-Frame Consensus.

---

## Real Data Characteristics (from 10-dataset analysis)

**CRITICAL**: These observations come from analyzing ALL 10 datasets and MUST guide every design decision.

### Dataset Overview
- **10 datasets**, all with exactly **300 frames** each
- **5 unique resolutions** (all 8-bit):
  - 358×433: volume_001, 002, 003
  - 269×512: volume_004
  - 338×344: volume_005, 006, 007
  - 274×512: volume_008, 009
  - 354×456: volume_010
- **Mean intensity**: 9.3–21.2 (out of 255), median ~12. Extremely low dynamic range.
- **Noise floor**: 0.0 across all datasets (many pixels are exactly zero).

### Pair Spacing (THE most critical parameter — CONFIRMED)
- **Global mean: 39.9 px**, std=6.0, median=40.0, range [30, 50]
- **Remarkably consistent across all resolutions**: per-dataset means range from 39.0 to 41.0 px
- The pair spacing does NOT vary with image resolution → it is a fixed property of the Bessel beam optics
- **Config range [30, 50] is CONFIRMED as correct**
- **Ideal distance for cost function: 40.0 px**

### Temporal Activity (calibrated from hand annotations — ALL 10 volumes)

**Data source**: User's frame-by-frame annotations of all 10 datasets, distinguishing paired vs single-spot neurons. Total: 314 annotation entries across 3000 frames.

**Active frame statistics**:
- **Active frame fraction**: 4%–42% across datasets (mean 16%, median 11%)
- **Wide variation**: volume_003 (42%) and volume_010 (39%) are very active; volume_002 (4%) and volume_007 (5%) are sparse
- **Event entries per dataset**: 12–108 (mean 31)
- **Entry durations**: 77% single-frame, 16% two-frame, 7% longer (max 44 frames in vol_003)
- **Simultaneous neurons**: mean 3.3, median 3, max 13 (in vol_002 and vol_010)

**Event duration interpretation**: Most annotation entries span 1 frame because different neurons fire in different frames. The user confirmed individual neuron events last 3–7 frames, but in annotations each snapshot is recorded separately. Multi-frame entries (2+ frames) indicate the same set of neurons persisting across frames.

### *** CRITICAL: Paired vs Single-Spot Neurons — MUCH WORSE THAN INITIALLY ESTIMATED ***

**This is the single most impactful finding for algorithm design.**

From ALL 10 datasets combined (1,294 neuron×frame observations):
- **48.4% paired** (626 instances — both spots visible)
- **51.6% single-spot** (668 instances — only one spot visible)

**Per-dataset breakdown**:

| Dataset | Paired | Single | Single% |
|---------|--------|--------|---------|
| vol_001 | 105 | 43 | **29%** ← best case for pairing |
| vol_002 | 31 | 38 | 55% |
| vol_003 | 166 | 27 | **14%** ← best case overall |
| vol_004 | 29 | 76 | **72%** |
| vol_005 | 17 | 39 | 70% |
| vol_006 | 32 | 53 | 62% |
| vol_007 | 20 | 18 | 47% |
| vol_008 | 27 | 92 | **77%** ← worst case for pairing |
| vol_009 | 20 | 33 | 62% |
| vol_010 | 179 | 249 | 58% |

**Key insight**: Volume_001 (the first analyzed) was the BEST case for pairing at 29% single. The true global single-spot rate is **52%** — HALF of all neuron observations show only one spot.

**FUNDAMENTAL ALGORITHM REDESIGN REQUIRED**:

The original design treated pairing as the "primary" path and single-spot as a "recovery" path. This is WRONG. With 52% of observations being single-spot, the algorithm needs **two co-equal detection paths**:

1. **Paired path** (~48%): Geometric + temporal pairing. High confidence. Works well for vol_001 and vol_003.

2. **Single-spot path** (~52%): Detect individual spots with:
   - Temporal dynamics: signal is transient (present in <50% of frames), with 3–7 frame event profile
   - Local SNR: spot must exceed local background significantly
   - Spatial association: check if position ±40px horizontally has a paired detection in ANY frame
   - If spatially associated with a known pair → **medium-high confidence**
   - If isolated with good temporal dynamics → **medium confidence**
   - A "pair-only" algorithm will miss **OVER HALF** of all events in most datasets.

3. **Cross-frame association**: A neuron that appears as paired in frame t but single in frame t+1 is the SAME neuron. The algorithm must track identities across frames using spatial proximity, not require pairing in every frame.

4. **Confidence model**:
   - Paired + temporal correlation: **high** (0.8–1.0)
   - Single-spot + spatial match to known pair location: **medium-high** (0.6–0.8)
   - Single-spot + good temporal dynamics + no spatial match: **medium** (0.4–0.6)
   - Single-spot + weak dynamics: **low** (0.2–0.4)

### Photobleaching (MANDATORY CORRECTION)
- **9 out of 10 datasets** show significant photobleaching:
  - volume_005: **−36.7%** (worst)
  - volume_007: −32.9%, volume_006: −31.3%, volume_010: −29.6%, volume_009: −29.1%
  - volume_001: −20.9%, volume_004: −20.2%, volume_002: −19.5%, volume_008: −16.1%
  - volume_003: −0.3% (only exception)
- **A mandatory photobleaching correction step must be added to Module 1**, BEFORE baseline estimation. Without it, the ΔF/F₀ baseline will be biased and later frames will have artificially depressed signals.
- Implementation: fit exponential or linear decay to the per-frame mean intensity, divide each frame by the fitted trend.

### Background Properties
- **Background mask returned ALL ZEROS** in every dataset — the detection criteria (high mean + low CV + high presence) are too strict for this 8-bit data where most pixels are near-zero. The background module needs redesign.
- **Non-uniformity is severe**: ranges from 2.5× (volume_008) to **14.6×** (volume_004) across the 4×4 FOV grid
- **Horizontal band/stripe artifacts** visible in EDI images — caused by the Bessel beam scanning pattern. The FOV has alternating bright/dark horizontal bands. These will generate false positives if not handled.
- **Bottom portion of FOV is often completely dark** (outside illumination cone) — must auto-detect and mask the valid illumination region.
- **REVISED approach**: Instead of the failed background mask, use:
  1. Illumination mask: threshold on the temporal mean image to exclude permanently dark regions
  2. Local background subtraction: large-kernel median filter (radius ≥ 50px) on each frame or on summary maps
  3. Band artifact suppression: horizontal high-pass filtering or notch filter on the EDI

### Spot Properties
- **Spot sigma**: median=2.0 px, mean=3.6 px, range [2.0, 15.0]
- Most spots are very small (sigma=2–4 px), much smaller than originally assumed (3–12)
- **Revised LoG detection**: `min_sigma=2.0`, `max_sigma=8.0` (not 12)
- **Spot diameter** ≈ 4–8 pixels for typical neurons

### Pair Search Selectivity Problem
- The pair search found similar numbers of pairs across ALL distance ranges:
  e.g., volume_001: [20-35]=858, [30-50]=1107, [45-70]=1262, [65-100]=1543
- **Pair counts increase monotonically with range width** → no clear peak at [30-50]
- **This means geometric pairing ALONE is not selective enough** — there are too many random coincidental pairings among hundreds of candidate spots
- **IMPLICATION**: Must add stronger constraints beyond geometry:
  1. **Temporal correlation**: both spots in a true pair must have correlated activation timing
  2. **Local SNR**: both spots must exceed local background by a significant margin
  3. **Reduce false spots first**: more aggressive pre-filtering before pairing

### Practical Implications for Algorithm Design
1. **Photobleaching correction is MANDATORY** (new Module 0, before everything else)
2. **Temporal validation uses 3–7 frame event window** — events have clear multi-frame dynamics (confirmed by annotations). Longer clusters (up to 21 frames) represent sequential firing of multiple neurons.
3. **DUAL DETECTION PATH is essential** — 52% of neuron observations are single-spot (ranges from 14% to 77% across datasets). A pair-only approach will miss OVER HALF of events in most datasets. Both paired and single-spot detection are co-equal paths.
4. **The EDI approach remains critical** — mean 16% of frames active (range 4–42%), but individual neurons occupy only ~0.1% of pixels. Temporal std map is essential for spatial localization.
5. **Background mask needs redesign** — use illumination masking + local background subtraction + horizontal band artifact suppression.
6. **Pairing must include temporal correlation** — geometric pairing alone is not selective. Both spots must show correlated temporal activation.
7. **Pair spacing is universal** — 40 px across all datasets and resolutions.
8. **Image sizes vary** — the pipeline must handle 5 different resolutions.
9. **Per-frame detection + multi-frame voting is viable** — with 4–42% of frames active and mean 3.3 neurons per event, there are ample detections for cross-frame consensus.
10. **Up to 9 neurons fire simultaneously** — the detection pipeline must handle dense overlapping events in a single frame.

---

## Tech Stack & Environment

- **Language**: Python 3.9+
- **Package manager**: pip (use `requirements.txt`, NOT conda)
- **Config**: YAML via `pyyaml` (all hyperparameters in `config/default.yaml`)
- **Testing**: pytest
- **Type hints**: Required on all public functions (use `numpy.typing.NDArray` for arrays)
- **Docstrings**: Google style, in English

### Core Dependencies

```
numpy>=1.23
scipy>=1.9
scikit-image>=0.20
scikit-learn>=1.2
tifffile>=2023.1
pyyaml>=6.0
pandas>=1.5
matplotlib>=3.6
tqdm>=4.64
```

### Optional Dependencies (separate `requirements-optional.txt`)

```
torch>=1.12        # DeepCAD denoising
napari>=0.4.17     # Interactive visualization
cellpose>=2.0      # Alternative DL segmentation
suite2p>=0.13      # Registration reference
```

---

## Directory Structure

```
bessel_neuron_seg/
├── CLAUDE.md                      # This file
├── requirements.txt
├── requirements-optional.txt
├── setup.py                       # pip install -e .
├── config/
│   └── default.yaml               # All hyperparameters (single source of truth)
├── src/
│   └── bessel_seg/
│       ├── __init__.py
│       ├── pipeline.py            # Main entry: run_pipeline()
│       ├── config.py              # Config loading & validation (dataclass-based)
│       ├── data_types.py          # All shared data structures
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── channel_extract.py
│       │   ├── registration.py
│       │   ├── photobleach.py     # NEW: mandatory photobleaching correction
│       │   └── denoise.py         # DeepCAD wrapper (graceful fallback if torch absent)
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── baseline.py
│       │   ├── delta_f.py
│       │   ├── summary_maps.py
│       │   └── background_mask.py # REDESIGNED: illumination mask + band suppression
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── enhanced_image.py
│       │   ├── blob_detect.py
│       │   └── adaptive_threshold.py
│       ├── pairing/
│       │   ├── __init__.py
│       │   ├── geometric_match.py
│       │   ├── hungarian_solver.py
│       │   └── orphan_handler.py
│       ├── fusion/
│       │   ├── __init__.py
│       │   ├── frame_quality.py
│       │   ├── spatial_cluster.py
│       │   └── temporal_validate.py
│       ├── refinement/
│       │   ├── __init__.py
│       │   ├── gaussian_fit.py
│       │   ├── roi_builder.py
│       │   └── confidence_score.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── overlay.py         # ROI overlay on images
│       │   ├── temporal_plot.py   # Time-series plots
│       │   └── summary_report.py  # HTML/PDF quality report
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py         # Precision, Recall, F1, MLE
│           └── annotation_io.py   # Load/save ground truth annotations
├── scripts/
│   ├── run_single.py              # Process one dataset (folder of frames or .tif stack)
│   ├── run_batch.py               # Process all 10 datasets
│   ├── visualize_results.py       # Interactive napari visualization
│   └── evaluate.py                # Run evaluation against annotations
├── tests/
│   ├── conftest.py                # Shared fixtures (synthetic data generators)
│   ├── test_preprocessing.py
│   ├── test_temporal.py
│   ├── test_detection.py
│   ├── test_pairing.py
│   ├── test_fusion.py
│   ├── test_refinement.py
│   └── test_pipeline_integration.py
├── notebooks/                     # Exploratory / demo only, NOT production code
│   ├── 01_data_exploration.ipynb
│   └── 02_parameter_tuning.ipynb
└── data/                          # NOT committed to git
    ├── raw/                       # Original .tif files
    ├── denoised/                  # DeepCAD output cache
    ├── annotations/               # Ground truth JSON files
    └── results/                   # Pipeline output
```

---

## Data Structures (`data_types.py`)

All inter-module data exchange uses these dataclasses. **Never pass raw dicts between modules.**

```python
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
        """Neuron center = midpoint of the pair. Returns (y, x)."""
        return ((self.left.y + self.right.y) / 2,
                (self.left.x + self.right.x) / 2)

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
```

---

## Config System (`config.py`)

Use `dataclasses` to define a typed config, loaded from YAML. **Do NOT use raw dict access for config values.**

```python
from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    baseline_percentile: int = 10
    registration_method: str = "phase_correlation"  # "phase_correlation" | "none"
    registration_upsample: int = 10                 # Sub-pixel precision factor
    photobleach_correction: str = "linear"          # "linear" | "exponential" | "none"
    # 9/10 datasets show 16-37% intensity decline. Correction is mandatory.

@dataclass
class DeepCADConfig:
    enabled: bool = True
    patch_xy: int = 150
    patch_t: int = 40
    epochs: int = 30
    batch_size: int = 1
    lr: float = 5e-5

@dataclass
class BackgroundConfig:
    cv_threshold: float = 0.15
    presence_ratio: float = 0.8
    mean_threshold_percentile: int = 85
    noise_floor_percentile: int = 5

@dataclass
class EDIConfig:
    sigma_weight: float = 0.4
    maxproj_weight: float = 0.4
    corr_weight: float = 0.2

@dataclass
class DetectionConfig:
    min_sigma: float = 2.0            # Real data median sigma = 2.0
    max_sigma: float = 8.0            # Real data: most spots sigma 2-4, cap at 8
    num_sigma: int = 8
    threshold: float = 0.03           # LoG response threshold (lowered from 0.05)
    overlap: float = 0.5
    local_snr_threshold: float = 1.5  # For adaptive local filtering

@dataclass
class PairingConfig:
    horizontal_dist_min: int = 30     # Measured min: 33.9 px; allow margin
    horizontal_dist_max: int = 50     # Measured max: 43.6 px; allow margin
    vertical_offset_max: int = 5      # Measured max: 3.2 px; allow margin
    intensity_ratio_range: tuple[float, float] = (0.25, 4.0)  # Real data shows asymmetry as low as 0.45
    scale_diff_max: float = 0.4
    cost_weights: dict = field(default_factory=lambda: {
        "distance": 0.4,              # Penalize deviation from mean spacing (40.0 px)
        "intensity": 0.3,
        "scale": 0.3
    })
    ideal_distance: float = 40.0      # Measured global mean across all 10 datasets

@dataclass
class ClusteringConfig:
    eps: float = 8.0
    min_samples: int = 3

@dataclass
class TemporalValidationConfig:
    # CORRECTED: Events last 3-7 frames with gradual rise and decay.
    min_peak_dff: float = 0.3
    min_rise_frames: int = 1          # At least 1 frame of increasing signal
    min_decay_frames: int = 1         # At least 1 frame of decreasing signal
    max_event_duration: int = 10      # Events > 10 frames are suspicious
    max_active_fraction: float = 0.5  # Reject if signal in >50% of frames (background)
    event_window: int = 7             # Sliding window size for event detection

@dataclass
class RefinementConfig:
    gaussian_fit_radius: int = 15
    top_k_frames: int = 10
    min_confidence: float = 0.3              # Discard neurons below this

@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    deepcad: DeepCADConfig = field(default_factory=DeepCADConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    edi: EDIConfig = field(default_factory=EDIConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    temporal_validation: TemporalValidationConfig = field(default_factory=TemporalValidationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
```

Provide a `load_config(yaml_path: str) -> PipelineConfig` function that loads YAML and merges with defaults.

---

## Module Implementation Specs

### Module 1: Preprocessing (`preprocessing/`)

#### `channel_extract.py`

```python
def extract_green_channel(tif_path: str | Path) -> NDArray[np.float32]:
    """
    Load a multi-frame RGB TIFF and extract the green channel.

    Args:
        tif_path: Path to the .tif file.

    Returns:
        Array of shape (T, H, W), dtype float32, values in original intensity range.
        Known data: H=358, W=433 per frame, RGB 24-bit (8-bit per channel).

    Implementation:
        - Use tifffile.imread()
        - Handle both (T, H, W, 3) and (T, H, W) shapes gracefully
        - If input is already grayscale, return as-is with a warning log
        - Convert to float32 immediately to avoid integer overflow in later steps
        - Validate: assert T >= 10, "Too few frames"
        - Log actual shape, dtype, and intensity range for debugging
    """
```

#### `registration.py`

```python
def rigid_register(
    stack: NDArray[np.float32],
    method: str = "phase_correlation",
    reference: str = "mean",
    upsample_factor: int = 10,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """
    Apply rigid (translation-only) registration to a time-series stack.

    Args:
        stack: (T, H, W) image stack.
        method: "phase_correlation" or "none".
        reference: "mean" (use temporal mean as ref) or "best_snr" (highest SNR frame).
        upsample_factor: Sub-pixel precision via phase_cross_correlation.

    Returns:
        registered_stack: (T, H, W) aligned stack.
        shifts: (T, 2) array of (dy, dx) shifts applied to each frame.

    Implementation:
        - Use skimage.registration.phase_cross_correlation for shift estimation
        - Use scipy.ndimage.shift with order=1 (bilinear) for applying shifts
        - If method == "none", return input unchanged with zero shifts
        - Log max shift magnitude; warn if > 5 pixels
    """
```

#### `photobleach.py` (**NEW — MANDATORY**)

```python
def correct_photobleaching(
    stack: NDArray[np.float32],
    method: str = "linear",
) -> NDArray[np.float32]:
    """
    Correct photobleaching by normalizing the per-frame mean intensity trend.

    CRITICAL: 9/10 real datasets show 16–37% intensity decline over 300 frames.
    Without correction, baseline F₀ estimation is biased and later frames
    will have artificially suppressed ΔF/F₀ values.

    Args:
        stack: (T, H, W) registered stack.
        method: "linear" (fit line to per-frame means, divide by trend)
                "exponential" (fit exp decay, divide by trend)
                "none" (skip)

    Returns:
        Corrected stack where per-frame mean intensity is approximately constant.

    Implementation:
        - Compute per-frame mean: means = np.mean(stack, axis=(1, 2))  # (T,)
        - If method == "linear":
            slope, intercept = np.polyfit(range(T), means, 1)
            trend = intercept + slope * np.arange(T)
        - If method == "exponential":
            Fit means = A * exp(-t/tau) + C via curve_fit
            trend = fitted curve
        - correction_factor = trend[0] / trend  # normalize so frame 0 is unchanged
        - return stack * correction_factor[:, np.newaxis, np.newaxis]
        - Log the relative change: (trend[-1] - trend[0]) / trend[0]
    """
```

#### `denoise.py`

```python
def deepcad_denoise(
    stack: NDArray[np.float32],
    config: DeepCADConfig,
) -> NDArray[np.float32]:
    """
    Apply DeepCAD self-supervised denoising.

    Args:
        stack: (T, H, W) registered stack.
        config: DeepCAD hyperparameters.

    Returns:
        Denoised stack, same shape.

    Implementation:
        - Try to import deepcad; if unavailable, log warning and return input unchanged
        - Save stack to temporary .tif, invoke DeepCAD training + inference
        - Cache denoised output to data/denoised/ keyed by input filename hash
        - If config.enabled is False, skip and return input
    """

def fallback_temporal_denoise(
    stack: NDArray[np.float32],
    window: int = 5,
) -> NDArray[np.float32]:
    """
    Simple temporal median filter as DeepCAD fallback.

    Implementation:
        - scipy.ndimage.median_filter with size=(window, 1, 1)
        - Preserves spatial structure, smooths temporal noise
    """
```

---

### Module 2: Temporal Analysis (`temporal/`)

#### `baseline.py`

```python
def estimate_baseline(
    stack: NDArray[np.float32],
    percentile: int = 10,
) -> NDArray[np.float32]:
    """
    Estimate baseline fluorescence F0 per pixel.

    Args:
        stack: (T, H, W)
        percentile: Low percentile to use (avoids activation frames).

    Returns:
        F0: (H, W) baseline image, float32.

    Implementation:
        - np.percentile(stack, percentile, axis=0)
        - Clip minimum to noise_floor (e.g., 1.0) to avoid division by near-zero later
    """
```

#### `delta_f.py`

```python
def compute_delta_f(
    stack: NDArray[np.float32],
    baseline: NDArray[np.float32],
    epsilon: float = 1.0,
) -> NDArray[np.float32]:
    """
    Compute ΔF/F0 normalized stack.

    Returns:
        dff: (T, H, W) float32 array. Values typically in [-1, +inf).

    Implementation:
        - dff = (stack - baseline[np.newaxis]) / np.maximum(baseline[np.newaxis], epsilon)
    """
```

#### `summary_maps.py`

```python
def compute_summary_maps(
    stack: NDArray[np.float32],
    dff: NDArray[np.float32],
) -> dict[str, NDArray[np.float32]]:
    """
    Compute temporal summary statistics maps.

    Returns dict with keys:
        - "sigma": temporal std of dff, shape (H, W)
        - "max_proj": temporal max of dff, shape (H, W)
        - "mean": temporal mean of raw stack, shape (H, W)
        - "cv": coefficient of variation = std(raw) / (mean(raw) + eps), shape (H, W)

    Implementation:
        - All computations along axis=0
        - Normalize each map to [0, 1] range for downstream fusion
    """
```

#### `background_mask.py` (**REDESIGNED — original approach returned all zeros on real data**)

```python
def generate_illumination_mask(
    stack: NDArray[np.float32],
    threshold_percentile: int = 5,
) -> NDArray[np.bool_]:
    """
    Identify the valid illumination region (exclude permanently dark areas).

    Real data shows large dark regions (bottom of FOV, corners) that are
    outside the illumination cone. These must be excluded before detection.

    Args:
        stack: (T, H, W)
        threshold_percentile: Pixels with temporal mean below this percentile
                              of the overall mean image are considered outside FOV.

    Returns:
        mask: (H, W) boolean, True = VALID region (illuminated).

    Implementation:
        - mean_img = np.mean(stack, axis=0)
        - threshold = np.percentile(mean_img, threshold_percentile)
        - mask = mean_img > max(threshold, 1.0)
        - Apply morphological opening (disk r=5) to clean up edges
    """

def suppress_band_artifacts(
    image: NDArray[np.float32],
    orientation: str = "horizontal",
) -> NDArray[np.float32]:
    """
    Suppress horizontal band/stripe artifacts in the EDI.

    Real data EDI shows strong horizontal bands from the Bessel beam scanning
    pattern. These create false spots along band boundaries.

    Implementation:
        - Compute row-wise mean profile: row_profile = np.mean(image, axis=1)
        - Smooth the row profile: row_smooth = gaussian_filter1d(row_profile, sigma=20)
        - Subtract: corrected = image - row_smooth[:, np.newaxis]
        - Clip negatives to zero
    """
```

---

### Module 3: Detection (`detection/`)

#### `enhanced_image.py`

```python
def build_enhanced_detection_image(
    summary_maps: dict[str, NDArray[np.float32]],
    background_mask: NDArray[np.bool_],
    config: EDIConfig,
) -> NDArray[np.float32]:
    """
    Build the Enhanced Detection Image (EDI) by fusing temporal summary maps.

    EDI = α * sigma_map + β * maxproj_map + γ * corr_map (if available)

    Implementation:
        - Each input map must already be normalized to [0, 1]
        - Apply background_mask: set EDI[background_mask] = 0
        - Final EDI normalized to [0, 1]
    """
```

#### `blob_detect.py`

```python
def detect_spots(
    image: NDArray[np.float32],
    config: DetectionConfig,
) -> list[Spot]:
    """
    Detect candidate spots using multi-scale LoG.

    Implementation:
        - Use skimage.feature.blob_log on the input image
        - blob_log returns (y, x, sigma) per detection
        - For each blob, measure intensity = image[int(y), int(x)]
        - Return list of Spot objects
    """

def detect_spots_per_frame(
    dff_stack: NDArray[np.float32],
    background_mask: NDArray[np.bool_],
    config: DetectionConfig,
    quality_threshold: float = 0.3,
) -> dict[int, list[Spot]]:
    """
    Run spot detection on each frame independently (alternative to EDI-based).

    Returns:
        Dict mapping frame_idx -> list of Spots detected in that frame.

    Implementation:
        - Skip frames where np.max(dff_stack[t]) < quality_threshold
        - For each valid frame: mask background, run detect_spots
        - Set spot.frame_idx for each detection
    """
```

#### `adaptive_threshold.py`

```python
def filter_spots_adaptive(
    spots: list[Spot],
    image: NDArray[np.float32],
    config: DetectionConfig,
) -> list[Spot]:
    """
    Filter spots by local SNR.

    For each spot:
        - Inner region: circle of radius 2*sigma
        - Outer ring: annulus from 2*sigma to 4*sigma
        - local_snr = (mean_inner - mean_outer) / std_outer
        - Keep if local_snr > config.local_snr_threshold

    Also filter spots too close to image borders (within max_sigma pixels).
    """
```

---

### Module 4: Pairing (`pairing/`)

#### `geometric_match.py`

```python
def find_candidate_pairs(
    spots: list[Spot],
    config: PairingConfig,
) -> list[tuple[int, int, float]]:
    """
    Find all geometrically valid spot pairs.

    For spots[i] and spots[j] where spots[i].x < spots[j].x:
        - horizontal_dist = spots[j].x - spots[i].x
        - vertical_offset = abs(spots[j].y - spots[i].y)
        - Check: horizontal_dist in [dist_min, dist_max]
        - Check: vertical_offset <= vertical_offset_max
        - Check: intensity ratio within range
        - Check: scale difference within max

    Returns:
        List of (i, j, cost) tuples. Cost is weighted combination of:
            - Normalized distance deviation from ideal_distance (default 40.0 px)
            - Intensity ratio deviation from 1.0
            - Scale difference

    Implementation:
        - Use scipy.spatial.KDTree for efficient neighbor search
        - Query with max_distance = sqrt(dist_max^2 + vertical_offset_max^2)
        - Then filter by individual constraints
    """
```

#### `hungarian_solver.py`

```python
def solve_optimal_pairing(
    spots: list[Spot],
    candidate_pairs: list[tuple[int, int, float]],
) -> tuple[list[SpotPair], list[Spot]]:
    """
    Find globally optimal 1-to-1 spot pairing using the Hungarian algorithm.

    Returns:
        paired: List of SpotPair objects.
        orphans: List of Spot objects that were not paired.

    Implementation:
        - Build a cost matrix from candidate_pairs
        - Use scipy.optimize.linear_sum_assignment
        - Threshold: reject assignments where cost > max_acceptable_cost
        - Remaining unmatched spots → orphans list
        - In SpotPair, ensure left.x < right.x
    """
```

#### `orphan_handler.py` (**REDESIGNED — handles the 52% single-spot neurons**)

```python
def rescue_orphans_across_frames(
    orphans_per_frame: dict[int, list[Spot]],
    pairs_per_frame: dict[int, list[SpotPair]],
    config: PairingConfig,
) -> list[SpotPair]:
    """
    Attempt to rescue orphan spots by cross-referencing with adjacent frames.

    For each orphan in frame t:
        - Search frames [t-2, t+2] for paired neurons near the orphan location
        - If a paired neuron exists nearby (< eps pixels), the orphan is likely
          a degraded version → create a low-confidence SpotPair using the
          known pair geometry from the adjacent frame as template.

    Returns:
        Additional SpotPair objects rescued from orphans.
    """

def classify_single_spot_neurons(
    orphans: list[Spot],
    known_pairs: list[SpotPair],
    dff: NDArray[np.float32],
    config: PairingConfig,
) -> list[NeuronROI]:
    """
    Classify remaining unpaired single spots as potential single-spot neurons.

    CRITICAL: 52% of neuron observations in real data show only ONE spot.
    These must not be discarded.

    Classification tiers:
    1. HIGH confidence: orphan is within 25px of a known paired neuron center,
       AND has correlated temporal dynamics → same neuron, one lobe suppressed.
       Assign the neuron center from the known pair.

    2. MEDIUM confidence: orphan has clear transient temporal dynamics
       (event-like: rises, peaks, decays over 3-7 frames), not persistent,
       but no nearby paired neuron found.
       → Likely a real neuron whose partner spot is always below noise threshold.
       Assign center at spot location (not pair midpoint).

    3. LOW confidence / REJECT: orphan is persistent (present in >50% of frames),
       or has no temporal dynamics, or is in a known artifact region.
       → Background artifact, discard.

    Returns:
        List of NeuronROI objects for single-spot neurons with confidence tier.
    """
```

---

### Module 5: Fusion (`fusion/`)

#### `frame_quality.py`

```python
def compute_frame_quality(
    stack: NDArray[np.float32],
    spots_per_frame: dict[int, list[Spot]],
    pairs_per_frame: dict[int, list[SpotPair]],
) -> list[FrameQuality]:
    """
    Score each frame's quality.

    Metrics:
        - SNR: np.mean(signal_region) / np.std(background_region) for that frame
        - Sharpness: np.var(laplacian(frame))  [scipy.ndimage.laplace]
        - PairRate: len(pairs) / max(len(spots), 1) for that frame

    overall_score = 0.4 * norm(SNR) + 0.3 * norm(Sharpness) + 0.3 * PairRate

    Normalize SNR and Sharpness across frames to [0, 1] before combining.
    """
```

#### `spatial_cluster.py`

```python
def cluster_neuron_detections(
    all_pairs: list[SpotPair],
    config: ClusteringConfig,
) -> list[list[SpotPair]]:
    """
    Cluster SpotPairs across frames by spatial proximity of their centers.

    Implementation:
        - Extract center (y, x) from each SpotPair
        - Run DBSCAN(eps=config.eps, min_samples=config.min_samples)
        - Each cluster = one neuron candidate
        - Return list of clusters; each cluster is a list of SpotPairs from different frames
        - Noise points (DBSCAN label = -1) are discarded

    IMPORTANT: Ignore frame index in clustering — neurons are spatially stable.
    """
```

#### `temporal_validate.py`

```python
def validate_calcium_dynamics(
    clusters: list[list[SpotPair]],
    dff: NDArray[np.float32],
    config: TemporalValidationConfig,
) -> list[list[SpotPair]]:
    """
    Validate neuron candidates by checking temporal signal has event-like dynamics.

    CORRECTED: Events last 3-7 frames with gradual rise and decay (user confirmed).
    The analysis tool's "1-frame events" was a bug caused by global-mean detection.

    For each cluster:
        1. Compute the weighted center position
        2. Extract ΔF/F0 time trace from a circular ROI (r=6 px) around center
        3. Detect events using a sliding window of config.event_window frames:
           - Find peaks where trace > config.min_peak_dff
           - Check for at least min_rise_frames of increase before peak
           - Check for at least min_decay_frames of decrease after peak
        4. Compute active_fraction = fraction of frames where trace > 0.5 * peak
        5. REJECT if:
           - active_fraction > config.max_active_fraction (persistent = background)
           - No events detected at all (never fires, likely noise)
           - All "events" exceed max_event_duration (artifact, not neural)
        6. ACCEPT if at least one valid event with rise-peak-decay structure

    Returns:
        Filtered list of clusters that pass validation.
    """
```

---

### Module 6: Refinement (`refinement/`)

#### `gaussian_fit.py`

```python
def fit_spot_gaussian(
    image: NDArray[np.float32],
    spot_center: tuple[float, float],
    radius: int = 15,
) -> tuple[float, float, float, float]:
    """
    Fit a 2D Gaussian to a spot and return refined parameters.

    Model: f(x, y) = offset + amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))

    Args:
        image: 2D image (single frame or temporal average of top-K frames).
        spot_center: (y, x) initial center estimate.
        radius: Half-size of the fitting window.

    Returns:
        (y0, x0, sigma, amplitude) — refined parameters.

    Implementation:
        - Extract local patch: image[y-r:y+r, x-r:x+r]
        - Build meshgrid for (Y, X) coordinates
        - scipy.optimize.curve_fit with bounds:
            - x0, y0: within ±radius of initial
            - sigma: [1.0, radius]
            - amplitude: [0, 2 * max(patch)]
            - offset: [0, median(patch)]
        - On fit failure (RuntimeError), return initial estimates with fallback sigma=5.0
    """
```

#### `roi_builder.py`

```python
def build_neuron_rois(
    clusters: list[list[SpotPair]],
    stack: NDArray[np.float32],
    dff: NDArray[np.float32],
    frame_qualities: list[FrameQuality],
    config: RefinementConfig,
) -> list[NeuronROI]:
    """
    Build final NeuronROI objects with fitted parameters and traces.

    For each cluster:
        1. Select top-K frames by FrameQuality.overall_score
        2. Compute weighted average position from SpotPairs in cluster
        3. Build average image from top-K frames
        4. Fit Gaussian to left and right spots on average image
        5. Build binary masks: pixels within 2*fitted_sigma of each spot center
        6. Extract temporal trace: for each frame, mean intensity within combined mask
        7. Assign neuron_id (sequential)

    Returns:
        List of NeuronROI objects.
    """
```

#### `confidence_score.py`

```python
def compute_confidence(
    cluster: list[SpotPair],
    frame_qualities: list[FrameQuality],
    temporal_trace: NDArray[np.float32],
    total_frames: int,
) -> float:
    """
    Compute confidence score in [0, 1] for a neuron.

    Components (weights sum to 1.0):
        - detection_fraction (0.25): len(cluster) / total_frames
        - pair_quality (0.25): mean of intensity_ratio and scale_similarity across pairs
        - temporal_fit (0.25): correlation of trace with ideal calcium template
        - spatial_consistency (0.25): 1 / (1 + std of center positions across pairs)

    Returns:
        Clipped to [0, 1].
    """
```

---

### Pipeline Orchestration (`pipeline.py`)

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_pipeline(
    data_path: str | Path,
    config: PipelineConfig | None = None,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    Main entry point. Processes one dataset (folder of frames or multi-frame .tif).

    Steps:
        1. Load config (default if not provided)
        2. Load data: folder-of-frames → (T, H, W) stack, extract green channel
        3. Module 1: register → **photobleach correct** → denoise
        4. Module 2: baseline → ΔF/F0 → summary maps → **illumination mask**
        5. Module 3: build EDI → **suppress band artifacts** → detect spots → adaptive filter
        6. Module 4: **PRIMARY PATH** — pair spots (geometry + temporal correlation) → resolve conflicts
        7. Module 4b: **SECONDARY PATH** — classify unpaired orphan spots → single-spot neurons
        8. Module 5: cluster across frames → **event-based temporal validation** (3-7 frame rise-decay)
        9. Module 6: Gaussian fit → build ROIs → confidence scores (tiered: paired/single_near/single_isolated)
        10. Save results to output_dir (if provided)

    Data loading:
        - If data_path is a directory: load folder-of-frames (natural sort)
        - If data_path is a .tif file: load multi-frame stack
        - All resolutions handled (269×512 to 358×433)

    Logging:
        - Log timing for each module
        - Log photobleaching correction magnitude
        - Log counts: frames, spots detected, pairs formed, clusters, final neurons
        - Log warnings for: low pair rate, no neurons found, extreme photobleaching
    """

def run_pipeline_dual_path(
    tif_path: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Enhanced pipeline with two detection paths merged:
        Path A: EDI-based detection (fast, one-shot)
        Path B: Per-frame detection + multi-frame voting (robust, slow)
    Merge: union of neurons from both paths, deduplicated by spatial proximity.

    Use this when Path A alone has low recall.
    """
```

---

## Configuration File (`config/default.yaml`)

```yaml
preprocessing:
  baseline_percentile: 10
  registration_method: "phase_correlation"
  registration_upsample: 10
  photobleach_correction: "linear"    # MANDATORY: 9/10 datasets lose 16-37% intensity

deepcad:
  enabled: true
  patch_xy: 150
  patch_t: 40
  epochs: 30
  batch_size: 1
  lr: 0.00005

background:
  # NOTE: CV/presence mask approach FAILED on real data (returned all zeros).
  # Use illumination masking + local subtraction instead.
  illumination_threshold_percentile: 5
  local_bg_radius: 50
  band_artifact_suppression: true

edi:
  sigma_weight: 0.4
  maxproj_weight: 0.4
  corr_weight: 0.2

detection:
  min_sigma: 2.0              # Real: median spot sigma=2.0
  max_sigma: 8.0              # Real: most spots sigma 2-4
  num_sigma: 8
  threshold: 0.03
  overlap: 0.5
  local_snr_threshold: 1.5

pairing:
  horizontal_dist_min: 30
  horizontal_dist_max: 50
  ideal_distance: 40.0        # Global mean across all 10 datasets
  vertical_offset_max: 5
  intensity_ratio_range: [0.25, 4.0]
  scale_diff_max: 0.4
  temporal_correlation_min: 0.3   # Both spots must co-activate
  cost_weights:
    distance: 0.3
    intensity: 0.2
    scale: 0.2
    temporal_correlation: 0.3

clustering:
  eps: 8.0
  min_samples: 3              # Restored: ~20-30 events × 3-7 frames = many detections per neuron

temporal_validation:
  # Calibrated from volume_001 annotations:
  # Individual entries: 1-7 frames. Event clusters: 1-21 frames, median=4.
  min_peak_dff: 0.3
  min_rise_frames: 1
  min_decay_frames: 1
  max_event_duration: 10      # Single neuron event; clusters can be longer
  max_active_fraction: 0.5    # Reject if active >50% of frames
  event_window: 7             # Matches observed typical event span

refinement:
  gaussian_fit_radius: 10     # Spots are small (sigma 2-4)
  top_k_frames: 10            # Pick best frames from ~20-30 events × 3-7 frames
  min_confidence: 0.3
```

---

## Testing Strategy

### Synthetic Data Fixtures (`tests/conftest.py`)

Generate synthetic test data that mimics real Bessel beam images:

```python
@pytest.fixture
def synthetic_stack():
    """
    Generate a (300, 358, 433) synthetic stack mimicking real data with:
    - 4 neurons total:
      - 2 paired neurons (both spots visible, spacing=40 px)
      - 1 neuron that alternates between paired and single-spot across events
      - 1 neuron that is always single-spot (partner spot suppressed)
    - Gaussian spot profiles (sigma=3, small like real data)
    - ~17 event clusters, 3-7 frame calcium events with gradual rise-peak-decay
    - ~22% of frames active, mean 2.7 neurons per active frame
    - Additive Gaussian noise (SNR ~50)
    - Spatially non-uniform background (2-8× non-uniformity, horizontal bands)
    - Linear photobleaching: -25% intensity drop over recording
    - One persistent background blob (present in all frames, for rejection testing)
    - Dark region at bottom of FOV (outside illumination cone)
    """
```

### Test Guidelines

- **Unit tests**: Each module function has ≥1 test. Test edge cases: empty input, single frame, all-noise frame.
- **Integration test**: `test_pipeline_integration.py` runs the full pipeline on synthetic data and checks that all 3 synthetic neurons are found with confidence > 0.5.
- **Determinism**: Set `np.random.seed(42)` in all tests involving randomness.
- **Performance**: Mark slow tests with `@pytest.mark.slow`. Default `pytest` run skips these.

---

## Scripts

### `scripts/run_single.py`

```python
"""
Usage: python scripts/run_single.py path/to/data.tif [--config config/default.yaml] [--output results/]
"""
# Use argparse. Load config, call run_pipeline, save results, print summary.
```

### `scripts/run_batch.py`

```python
"""
Usage: python scripts/run_batch.py data/raw/ [--config config/default.yaml] [--output data/results/]
"""
# Glob all .tif files, process sequentially (or with multiprocessing.Pool), aggregate stats.
```

---

## Output Format

### Per-dataset output directory structure:

```
results/dataset_001/
├── neurons.json           # List of {neuron_id, center_y, center_x, left_spot, right_spot, confidence, ...}
├── roi_masks.tif          # (N, H, W) uint8 label image, 0=background, i=neuron_i
├── traces.csv             # Columns: frame_idx, neuron_0, neuron_1, ..., neuron_N
├── frame_quality.csv      # Columns: frame_idx, snr, sharpness, pair_rate, overall_score
├── summary_maps.npz       # sigma, max_proj, cv, edi, background_mask
└── report.html            # Visual quality report with overlays and plots
```

### `neurons.json` schema:

```json
[
  {
    "neuron_id": 0,
    "center_y": 128.5,
    "center_x": 200.3,
    "left_spot": {"y": 128.2, "x": 179.1, "sigma": 5.3, "radius": 7.8},
    "right_spot": {"y": 128.8, "x": 221.5, "sigma": 5.1, "radius": 7.5},
    "detection_type": "paired",
    "confidence": 0.87,
    "detection_count": 45,
    "total_frames": 300
  },
  {
    "neuron_id": 1,
    "center_y": 250.0,
    "center_x": 310.5,
    "left_spot": {"y": 250.1, "x": 310.5, "sigma": 3.2, "radius": 5.0},
    "right_spot": null,
    "detection_type": "single_near_pair",
    "confidence": 0.52,
    "detection_count": 12,
    "total_frames": 300
  }
]
```

---

## Coding Conventions

1. **Array shapes**: Always document as comments: `# (T, H, W)`, `# (N, 2)` etc.
2. **Coordinate order**: Use `(y, x)` consistently (image convention, row-major). Never mix with `(x, y)`.
3. **Logging**: Use `logging` module, NOT print(). Level: INFO for progress, DEBUG for details, WARNING for issues.
4. **Error handling**: Raise `ValueError` for bad inputs. Use try/except around `curve_fit` and other optimization calls.
5. **No global state**: All state flows through function arguments and return values.
6. **Memory**: For large stacks, process frame-by-frame where possible. Document memory-intensive operations.
7. **Imports**: Group as stdlib → third-party → local. Use absolute imports (`from bessel_seg.data_types import Spot`).
8. **Magic numbers**: Every numerical constant must come from config or be defined as a module-level constant with a comment explaining its origin.
9. **Progress bars**: Use `tqdm` for any loop over frames or neurons that may take > 5 seconds.
10. **Reproducibility**: All random operations must accept an optional `seed` parameter.

---

## Critical Implementation Notes

### Bessel Beam Physics Context

The dual-spot pattern arises from the **annular side-lobe structure** of Bessel beams. The two spots are symmetric about the neuron's true center, always aligned along the **horizontal (x) axis** of the scan. **Confirmed spacing: 40.0 px (mean), range 30–50 px, IDENTICAL across all 10 datasets and all 5 image resolutions**. This is a fixed optical property of the beam, not dependent on image magnification or FOV size.

### Key Pitfalls to Avoid

1. **Do NOT use Suite2p/CaImAn directly** — they assume single-spot PSF and will merge or miss paired spots.
2. **Do NOT threshold on raw intensity** — background can be brighter than dim neurons. Always use ΔF/F0 or temporal statistics. Real data shows mean intensity ~11/255 with neuron signals only 30–50 counts above local background.
3. **Do NOT assume all neurons are active** — some may never fire during the recording. The EDI path captures these via temporal variance (from noise fluctuations) but with lower confidence.
4. **Pair spacing is UNIVERSAL at 40 px** — confirmed identical across all 10 datasets and all 5 resolutions. Use configurable range [30, 50] but do not expect it to vary per dataset.
5. **Watch for edge effects** — spots near image borders may have truncated profiles. Exclude spots within `max_sigma` pixels of borders.
6. **Integer vs. float coordinates** — Spot centers from LoG are float (sub-pixel). Keep them as float through pairing and clustering. Only convert to int for array indexing.
7. **Do NOT use global intensity thresholds** — background non-uniformity ranges from 2.5× to 14.6× across the FOV. Always subtract local background before intensity comparisons.
8. **Do NOT rely on single-frame detection** — individual neurons occupy only ~0.1% of the FOV. The EDI (multi-frame fusion) is essential for reliable detection.
9. **All data is 8-bit** (confirmed across all 10 datasets). Convert to float32 immediately upon loading. Resolutions vary: 269×512, 274×512, 338×344, 354×456, 358×433.
10. **Geometric pairing alone is NOT selective** — pair counts increase monotonically across distance ranges. Must add temporal correlation between paired spots.
11. **Correct photobleaching BEFORE baseline estimation** — 9/10 datasets lose 16-37% intensity. Without correction, ΔF/F₀ is biased.

### Available Ground Truth Annotations

Hand-annotated firing event records exist for all 10 datasets. Format per file:
```
[frame_start-frame_end]:N_paired(paired)+N_single(single)
```
Example: `[147]:4(paired)+5(single)` = frame 147 has 4 paired-spot and 5 single-spot firing neurons.

These annotations can be used for:
1. **Recall evaluation**: does the algorithm detect events in the annotated frames?
2. **Temporal parameter tuning**: calibrate event detection window, ΔF/F₀ threshold
3. **Paired vs single ratio validation**: does the algorithm's pair/single ratio match ~48%/52% globally (varies 14–77% single per dataset)?
4. **NOT for precision evaluation** — annotations may miss some events; absence ≠ no activity

Key statistics from volume_001 (300 frames):
- 44 entries, 66 active frames (22%), 17 event clusters
- mean 2.7 neurons per active frame, max 9
- 48% paired, 52% single-spot observations (global; varies 14–77% single per dataset)

### Performance Targets

- Processing one 300-frame dataset: < 60 seconds (excluding DeepCAD)
- DeepCAD denoising: ~3–10 minutes per dataset (GPU) — cache results
- Memory: < 1 GB for a 300-frame, 512×274 dataset (largest resolution)
