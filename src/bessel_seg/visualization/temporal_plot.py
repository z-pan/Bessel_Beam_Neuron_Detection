"""ΔF/F₀ temporal trace plots for detected neurons."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from bessel_seg.data_types import NeuronROI

# Colour scheme consistent with overlay.py
_COLOURS: dict[str, str] = {
    "paired": "#4682FF",
    "single_near_pair": "#FFA500",
    "single_isolated": "#32CD32",
}
_FALLBACK_COLOUR = "#AAAAAA"


def plot_neuron_traces(
    neurons: list[NeuronROI],
    save_path: Optional[str | Path] = None,
    title: str = "Neuron ΔF/F₀ Traces",
    figsize_per_row: tuple[float, float] = (12.0, 1.6),
    dff_range: Optional[tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Plot ΔF/F₀ temporal traces for all neurons in a stacked layout.

    Args:
        neurons: List of NeuronROI objects.  Neurons without a temporal_trace
                 are skipped.
        save_path: If provided, save the figure to this path (PNG/PDF/SVG).
        title: Figure title.
        figsize_per_row: (width, height_per_neuron) in inches.
        dff_range: Optional fixed y-axis range (min, max).  Auto-scaled if None.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to neurons that have a trace
    plottable = [r for r in neurons if r.temporal_trace is not None]
    n = len(plottable)

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(figsize_per_row[0], 2.0))
        ax.text(0.5, 0.5, "No temporal traces available.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        if save_path:
            fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        return fig

    fig_height = figsize_per_row[1] * n + 1.0
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_row[0], fig_height),
        sharex=True,
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, y=1.01)

    for ax, roi in zip(axes, plottable):
        trace: NDArray[np.float32] = roi.temporal_trace  # (T,)
        T = len(trace)
        t = np.arange(T)
        colour = _COLOURS.get(roi.detection_type, _FALLBACK_COLOUR)

        ax.fill_between(t, 0, trace, where=trace > 0,
                        alpha=0.35, color=colour, interpolate=True)
        ax.plot(t, trace, color=colour, linewidth=0.9, label=roi.detection_type)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

        if dff_range is not None:
            ax.set_ylim(*dff_range)

        label = (
            f"N{roi.neuron_id}  ({roi.detection_type})"
            f"  conf={roi.confidence:.2f}"
            f"  ({roi.detection_count}/{roi.total_frames} frames)"
        )
        ax.set_ylabel("ΔF/F₀", fontsize=7)
        ax.set_title(label, fontsize=8, loc="left", pad=2)
        ax.tick_params(axis="both", labelsize=7)

        # Annotate peak
        peak_idx = int(np.argmax(trace))
        peak_val = float(trace[peak_idx])
        if peak_val > 0.05:
            ax.annotate(
                f"{peak_val:.2f}",
                xy=(peak_idx, peak_val),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=6,
                color=colour,
            )

    axes[-1].set_xlabel("Frame", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)

    return fig


def plot_frame_quality(
    frame_qualities: list,
    save_path: Optional[str | Path] = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Plot per-frame quality metrics.

    Args:
        frame_qualities: List of FrameQuality objects.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frame_qualities:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No frame quality data.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    indices = [fq.frame_idx for fq in frame_qualities]
    scores = [fq.overall_score for fq in frame_qualities]
    snrs = [fq.snr for fq in frame_qualities]
    pair_rates = [fq.pair_rate for fq in frame_qualities]

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    fig.suptitle("Per-frame Quality Metrics", fontsize=11)

    axes[0].plot(indices, scores, color="#1f77b4", linewidth=0.9)
    axes[0].set_ylabel("Overall\nScore", fontsize=8)

    axes[1].plot(indices, snrs, color="#ff7f0e", linewidth=0.9)
    axes[1].set_ylabel("SNR", fontsize=8)

    axes[2].plot(indices, pair_rates, color="#2ca02c", linewidth=0.9)
    axes[2].set_ylabel("Pair Rate", fontsize=8)
    axes[2].set_xlabel("Frame", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)

    return fig
