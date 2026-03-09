"""Configuration system for the Bessel beam neuron segmentation pipeline.

All hyperparameters are defined as typed dataclasses and loaded from YAML.
Never use raw dict access for config values — always use the typed attributes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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
    # Redesigned fields: illumination masking + band suppression
    # (original CV/presence mask returned all zeros on real data)
    illumination_threshold_percentile: int = 5
    local_bg_radius: int = 50
    band_artifact_suppression: bool = True


@dataclass
class EDIConfig:
    sigma_weight: float = 0.4
    maxproj_weight: float = 0.4
    corr_weight: float = 0.2


@dataclass
class DetectionConfig:
    min_sigma: float = 1.0            # Measured p10 sigma = 1.22
    max_sigma: float = 5.0            # Measured p90 sigma = 4.28
    num_sigma: int = 8
    threshold: float = 0.03           # LoG response threshold
    overlap: float = 0.5
    local_snr_threshold: float = 0.4  # Measured: paired p10 SNR = 0.41


@dataclass
class PairingConfig:
    horizontal_dist_min: int = 25     # Measured p10 = 29.8; 25 allows margin
    horizontal_dist_max: int = 60     # Measured p90 = 55.0; 60 allows margin
    vertical_offset_max: int = 5      # Measured max: 3.2 px; allow margin
    intensity_ratio_range: tuple[float, float] = (0.25, 4.0)  # Real data shows asymmetry as low as 0.45
    scale_diff_max: float = 0.4
    cost_weights: dict = field(default_factory=lambda: {
        "distance": 0.4,              # Penalize deviation from median spacing (43.0 px)
        "intensity": 0.3,
        "scale": 0.3
    })
    ideal_distance: float = 43.0      # Measured median from 689 confirmed pairs
    # Temporal correlation: both spots in a true pair must co-activate
    temporal_correlation_min: float = 0.3


@dataclass
class ClusteringConfig:
    eps: float = 8.0
    min_samples: int = 3


@dataclass
class TemporalValidationConfig:
    # CORRECTED: Events last 3-7 frames with gradual rise and decay.
    min_peak_dff: float = 0.20         # Measured: paired p10 ΔF/F₀ = 0.206
    min_rise_frames: int = 1          # At least 1 frame of increasing signal
    min_decay_frames: int = 1         # At least 1 frame of decreasing signal
    max_event_duration: int = 10      # Events > 10 frames are suspicious
    max_active_fraction: float = 0.5  # Reject if signal in >50% of frames (background)
    event_window: int = 7             # Sliding window size for event detection


@dataclass
class RefinementConfig:
    gaussian_fit_radius: int = 8       # Spots are small (median σ=2.5)
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


# ---------------------------------------------------------------------------
# Mapping from YAML top-level key → (dataclass type, attribute name on PipelineConfig)
# ---------------------------------------------------------------------------
_SECTION_MAP: dict[str, tuple[type, str]] = {
    "preprocessing": (PreprocessingConfig, "preprocessing"),
    "deepcad": (DeepCADConfig, "deepcad"),
    "background": (BackgroundConfig, "background"),
    "edi": (EDIConfig, "edi"),
    "detection": (DetectionConfig, "detection"),
    "pairing": (PairingConfig, "pairing"),
    "clustering": (ClusteringConfig, "clustering"),
    "temporal_validation": (TemporalValidationConfig, "temporal_validation"),
    "refinement": (RefinementConfig, "refinement"),
}


def _merge_section(default_obj: Any, overrides: dict) -> Any:
    """Return a new dataclass instance merging *overrides* onto *default_obj*.

    Args:
        default_obj: A dataclass instance with default values.
        overrides: Dict of field-name → value loaded from YAML.

    Returns:
        New dataclass instance of the same type.
    """
    cls = type(default_obj)
    # Start from current defaults as a dict
    current: dict[str, Any] = {
        f: getattr(default_obj, f)
        for f in default_obj.__dataclass_fields__  # type: ignore[attr-defined]
    }
    for key, value in overrides.items():
        if key not in current:
            logger.warning("Unknown config key '%s' in section '%s' — ignored.", key, cls.__name__)
            continue
        # Special-case: intensity_ratio_range in PairingConfig comes from YAML as a list
        if key == "intensity_ratio_range" and isinstance(value, (list, tuple)):
            current[key] = tuple(value)
        elif key == "cost_weights" and isinstance(value, dict):
            # Merge cost_weights dict instead of replacing entirely
            merged_weights: dict = dict(current[key])
            merged_weights.update(value)
            current[key] = merged_weights
        else:
            current[key] = value
    return cls(**current)


def load_config(yaml_path: str | Path) -> PipelineConfig:
    """Load a YAML config file and merge it with dataclass defaults.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        PipelineConfig with values from the YAML file, falling back to
        dataclass defaults for any key not present in the file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file cannot be parsed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    logger.info("Loading config from %s", yaml_path)

    with yaml_path.open("r") as fh:
        try:
            raw: dict = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config at {yaml_path}: {exc}") from exc

    # Build default config first
    config = PipelineConfig()

    # Merge each recognised top-level section
    for section_key, (section_cls, attr_name) in _SECTION_MAP.items():
        if section_key in raw:
            section_overrides = raw[section_key]
            if not isinstance(section_overrides, dict):
                logger.warning(
                    "Config section '%s' is not a dict (got %s) — skipped.",
                    section_key,
                    type(section_overrides).__name__,
                )
                continue
            default_section = getattr(config, attr_name)
            merged_section = _merge_section(default_section, section_overrides)
            object.__setattr__(config, attr_name, merged_section)

    # Warn about any unknown top-level keys
    for key in raw:
        if key not in _SECTION_MAP:
            logger.warning("Unknown top-level config key '%s' — ignored.", key)

    logger.debug("Final config: %s", config)
    return config
