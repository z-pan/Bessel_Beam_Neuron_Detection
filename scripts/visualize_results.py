#!/usr/bin/env python3
"""Visualize pipeline results: ROI overlays, temporal traces, summary maps.

Usage
-----
.. code-block:: bash

    # Static matplotlib output (always available)
    python scripts/visualize_results.py results/dataset_001/

    # Interactive napari viewer (requires napari)
    python scripts/visualize_results.py results/dataset_001/ --napari --stack data.tif

    # Save overlay image without displaying
    python scripts/visualize_results.py results/dataset_001/ --save overlay.png

Arguments
---------
result_dir
    Path to a pipeline output directory containing neurons.json,
    summary_maps.npz, and optionally traces.csv and frame_quality.csv.

--stack
    Optional path to the original .tif stack (needed for napari overlay
    and frame-level overlay rendering).

--napari
    Launch an interactive napari viewer (requires the napari package).

--save
    Save the overlay image to a file instead of displaying it.

--frame
    Frame index for the overlay (default: use the EDI/max-projection map).
    Only used when --stack is provided.

--no-traces
    Skip plotting temporal traces.

--loglevel
    Python logging level: DEBUG, INFO (default), WARNING, ERROR.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Bessel beam neuron segmentation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="Path to pipeline output directory (contains neurons.json, summary_maps.npz, etc.).",
    )
    parser.add_argument(
        "--stack",
        type=str,
        default=None,
        help="Path to the original .tif stack for overlay rendering.",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Launch interactive napari viewer (requires napari package).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save overlay image to this path (PNG/TIFF) instead of displaying.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index for overlay (default: use EDI or max-projection).",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Skip temporal trace plotting.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def _load_neurons(result_dir: Path) -> list[dict]:
    """Load neurons.json from the result directory."""
    neurons_path = result_dir / "neurons.json"
    if not neurons_path.exists():
        raise FileNotFoundError(f"neurons.json not found in {result_dir}")
    with open(neurons_path) as f:
        return json.load(f)


def _load_summary_maps(result_dir: Path) -> dict[str, np.ndarray]:
    """Load summary_maps.npz from the result directory."""
    maps_path = result_dir / "summary_maps.npz"
    if not maps_path.exists():
        return {}
    data = np.load(str(maps_path))
    return {k: data[k] for k in data.files}


def _neurons_to_rois(neurons_data: list[dict]) -> list:
    """Convert JSON neuron dicts to NeuronROI objects for overlay rendering."""
    from bessel_seg.data_types import NeuronROI, Spot

    rois = []
    for nd in neurons_data:
        left_spot = None
        right_spot = None
        if nd.get("left_spot"):
            ls = nd["left_spot"]
            left_spot = Spot(
                y=ls["y"], x=ls["x"],
                sigma=ls.get("sigma", 2.5),
                intensity=ls.get("intensity", 1.0),
            )
        if nd.get("right_spot"):
            rs = nd["right_spot"]
            right_spot = Spot(
                y=rs["y"], x=rs["x"],
                sigma=rs.get("sigma", 2.5),
                intensity=rs.get("intensity", 1.0),
            )
        roi = NeuronROI(
            neuron_id=nd["neuron_id"],
            center_y=nd["center_y"],
            center_x=nd["center_x"],
            left_spot=left_spot,
            right_spot=right_spot,
            left_radius=nd.get("left_radius", 5.0),
            right_radius=nd.get("right_radius", 5.0),
            confidence=nd["confidence"],
            detection_type=nd["detection_type"],
            detection_count=nd.get("detection_count", 0),
            total_frames=nd.get("total_frames", 0),
        )
        rois.append(roi)
    return rois


def _visualize_matplotlib(
    result_dir: Path,
    neurons_data: list[dict],
    summary_maps: dict[str, np.ndarray],
    stack_path: Optional[str],
    frame_idx: Optional[int],
    save_path: Optional[str],
    show_traces: bool,
    logger: logging.Logger,
) -> None:
    """Render static matplotlib visualizations."""
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt

    from bessel_seg.visualization.overlay import draw_roi_overlay

    rois = _neurons_to_rois(neurons_data)

    # Determine the base image for the overlay
    base_image = None
    if stack_path and frame_idx is not None:
        import tifffile
        stack = tifffile.imread(stack_path)
        if stack.ndim == 3:
            base_image = stack[frame_idx].astype(np.float32)
        elif stack.ndim == 4:  # (T, H, W, C) — use green channel
            base_image = stack[frame_idx, :, :, 1].astype(np.float32)
        logger.info("Using frame %d from stack as overlay base.", frame_idx)
    elif "edi" in summary_maps:
        base_image = summary_maps["edi"]
        logger.info("Using EDI map as overlay base.")
    elif "max_proj" in summary_maps:
        base_image = summary_maps["max_proj"]
        logger.info("Using max-projection map as overlay base.")
    elif "sigma" in summary_maps:
        base_image = summary_maps["sigma"]
        logger.info("Using sigma map as overlay base.")

    if base_image is None:
        logger.error("No base image available for overlay. Provide --stack or ensure summary_maps.npz exists.")
        return

    # Draw overlay
    overlay = draw_roi_overlay(base_image, rois)

    # Build figure layout
    n_panels = 1
    if show_traces and (result_dir / "traces.csv").exists():
        n_panels = 2

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: ROI overlay
    axes[0].imshow(overlay)
    axes[0].set_title(f"ROI Overlay — {len(rois)} neurons", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Trace summary (optional)
    if n_panels > 1:
        import pandas as pd
        traces_df = pd.read_csv(result_dir / "traces.csv")
        neuron_cols = [c for c in traces_df.columns if c.startswith("neuron_")]
        for col in neuron_cols[:10]:  # limit to 10 traces for readability
            axes[1].plot(traces_df["frame_idx"], traces_df[col], linewidth=0.7, label=col)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("ΔF/F₀")
        axes[1].set_title("Temporal Traces (top 10)")
        axes[1].legend(fontsize=6, ncol=2)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Overlay saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)

    # Also display summary maps as a grid if available
    map_keys = [k for k in ("sigma", "max_proj", "mean", "cv", "edi") if k in summary_maps]
    if map_keys:
        n = len(map_keys)
        fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes2 = [axes2]
        for ax, key in zip(axes2, map_keys):
            ax.imshow(summary_maps[key], cmap="gray", aspect="auto")
            ax.set_title(key, fontsize=9)
            ax.axis("off")
        fig2.suptitle("Summary Maps", fontsize=11)
        fig2.tight_layout()

        if save_path:
            maps_save = str(Path(save_path).with_stem(Path(save_path).stem + "_maps"))
            fig2.savefig(maps_save, bbox_inches="tight", dpi=150)
            logger.info("Summary maps saved to %s", maps_save)
        else:
            plt.show()

        plt.close(fig2)


def _visualize_napari(
    result_dir: Path,
    neurons_data: list[dict],
    summary_maps: dict[str, np.ndarray],
    stack_path: Optional[str],
    logger: logging.Logger,
) -> None:
    """Launch interactive napari viewer with ROI overlays."""
    try:
        import napari
    except ImportError:
        logger.error(
            "napari is not installed. Install it with: pip install napari[all]  "
            "or use matplotlib mode (omit --napari)."
        )
        return

    from bessel_seg.visualization.overlay import make_roi_label_image

    rois = _neurons_to_rois(neurons_data)

    viewer = napari.Viewer(title="Bessel Beam Neuron Segmentation")

    # Add the original stack if provided
    if stack_path:
        import tifffile
        stack = tifffile.imread(stack_path).astype(np.float32)
        if stack.ndim == 4:  # (T, H, W, C)
            stack = stack[:, :, :, 1]  # green channel
        viewer.add_image(stack, name="Raw Stack", colormap="gray")
        H, W = stack.shape[-2], stack.shape[-1]
    elif "edi" in summary_maps:
        edi = summary_maps["edi"]
        viewer.add_image(edi, name="EDI", colormap="gray")
        H, W = edi.shape
    else:
        logger.error("No image data to display. Provide --stack or summary_maps.npz.")
        return

    # Add summary maps
    for key in ("sigma", "max_proj", "mean", "cv", "edi"):
        if key in summary_maps:
            viewer.add_image(
                summary_maps[key],
                name=f"Map: {key}",
                colormap="viridis",
                visible=False,
            )

    # Add ROI label image
    label_img = make_roi_label_image(rois, H, W)
    viewer.add_labels(label_img, name="ROI Labels", opacity=0.4)

    # Add neuron centres as points
    centres = np.array([[r.center_y, r.center_x] for r in rois])
    colours = []
    for r in rois:
        if r.detection_type == "paired":
            colours.append("blue")
        elif r.detection_type == "single_near_pair":
            colours.append("orange")
        else:
            colours.append("green")

    if len(centres) > 0:
        viewer.add_points(
            centres,
            name="Neuron Centres",
            face_color=colours,
            size=8,
            properties={
                "id": [r.neuron_id for r in rois],
                "type": [r.detection_type for r in rois],
                "confidence": [f"{r.confidence:.3f}" for r in rois],
            },
        )

    logger.info("Napari viewer launched with %d neurons.", len(rois))
    napari.run()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        logger.error("Result directory not found: %s", result_dir)
        return 1

    # Load data
    try:
        neurons_data = _load_neurons(result_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    summary_maps = _load_summary_maps(result_dir)

    n_total = len(neurons_data)
    n_paired = sum(1 for nd in neurons_data if nd.get("detection_type") == "paired")
    logger.info(
        "Loaded %d neurons (%d paired, %d single-spot) from %s",
        n_total, n_paired, n_total - n_paired, result_dir,
    )

    if args.napari:
        _visualize_napari(result_dir, neurons_data, summary_maps, args.stack, logger)
    else:
        _visualize_matplotlib(
            result_dir,
            neurons_data,
            summary_maps,
            args.stack,
            args.frame,
            args.save,
            show_traces=not args.no_traces,
            logger=logger,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
