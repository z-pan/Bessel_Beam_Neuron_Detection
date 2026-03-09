"""Generate an HTML summary report for a pipeline result."""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from bessel_seg.data_types import PipelineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib Figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _array_to_b64(arr: NDArray) -> str:
    """Encode a 2-D float array as a normalised grayscale PNG (base64)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    lo, hi = float(arr.min()), float(arr.max())
    vmin = lo if hi - lo > 1e-6 else 0.0
    vmax = hi if hi - lo > 1e-6 else 1.0
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.axis("off")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _img_tag(b64: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};"/>'


def _neuron_table_html(result: PipelineResult) -> str:
    rows = []
    for roi in result.neurons:
        rows.append(
            f"<tr>"
            f"<td>{roi.neuron_id}</td>"
            f"<td>{roi.center_y:.1f}</td>"
            f"<td>{roi.center_x:.1f}</td>"
            f"<td>{roi.detection_type}</td>"
            f"<td>{roi.confidence:.3f}</td>"
            f"<td>{roi.detection_count}/{roi.total_frames}</td>"
            f"</tr>"
        )
    header = (
        "<tr><th>ID</th><th>Y</th><th>X</th>"
        "<th>Type</th><th>Confidence</th><th>Detections</th></tr>"
    )
    return (
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px;'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _metadata_table_html(result: PipelineResult) -> str:
    rows = []
    for k, v in sorted(result.metadata.items()):
        rows.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
    return (
        "<table border='1' cellpadding='4' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px;'>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    result: PipelineResult,
    output_dir: str | Path,
    dataset_name: str = "dataset",
) -> Path:
    """Generate a self-contained HTML report for one pipeline run.

    The report includes:
    * Pipeline metadata table
    * Neuron summary table
    * Summary map images (sigma, max_proj, mean, cv, edi if present)
    * ΔF/F₀ trace plot for all neurons
    * Per-frame quality plot

    Args:
        result: PipelineResult from run_pipeline().
        output_dir: Directory where report.html is written.
        dataset_name: Dataset identifier for the report title.

    Returns:
        Path to the written report.html file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bessel_seg.visualization.temporal_plot import plot_neuron_traces, plot_frame_quality

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Summary map images ----
    map_sections = []
    for key in ("sigma", "max_proj", "mean", "cv", "edi"):
        arr = result.summary_maps.get(key)
        if arr is not None:
            b64 = _array_to_b64(arr)
            map_sections.append(
                f"<div style='display:inline-block;margin:6px;text-align:center;'>"
                f"<p style='font-size:12px;margin:2px;'>{key}</p>"
                f"{_img_tag(b64, key, '200px')}"
                f"</div>"
            )
    maps_html = "<div>" + "".join(map_sections) + "</div>" if map_sections else "<p>No maps.</p>"

    # ---- Trace plot ----
    trace_b64 = ""
    if any(r.temporal_trace is not None for r in result.neurons):
        fig_traces = plot_neuron_traces(result.neurons, title=f"{dataset_name} — ΔF/F₀ Traces")
        trace_b64 = _fig_to_b64(fig_traces)
        plt.close(fig_traces)

    # ---- Frame quality plot ----
    fq_b64 = ""
    if result.frame_qualities:
        fig_fq = plot_frame_quality(result.frame_qualities)
        fq_b64 = _fig_to_b64(fig_fq)
        plt.close(fig_fq)

    # ---- Background mask image ----
    bg_b64 = _array_to_b64(result.background_mask.astype(np.float32))

    # ---- Counts ----
    n_neurons = len(result.neurons)
    n_paired = sum(1 for r in result.neurons if r.detection_type == "paired")
    n_single = n_neurons - n_paired
    mean_conf = float(np.mean([r.confidence for r in result.neurons])) if result.neurons else 0.0

    # ---- Assemble HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Bessel Beam Pipeline Report — {dataset_name}</title>
<style>
  body {{font-family:Arial,sans-serif;margin:20px;background:#f9f9f9;}}
  h1 {{color:#333;border-bottom:2px solid #4682FF;padding-bottom:6px;}}
  h2 {{color:#555;margin-top:28px;}}
  .stat-box {{display:inline-block;background:#fff;border:1px solid #ccc;
              border-radius:6px;padding:10px 20px;margin:6px;text-align:center;}}
  .stat-num {{font-size:2em;font-weight:bold;color:#4682FF;}}
  .stat-lbl {{font-size:0.85em;color:#666;}}
  img {{border:1px solid #ddd;border-radius:4px;}}
</style>
</head>
<body>
<h1>Bessel Beam Pipeline Report</h1>
<p><b>Dataset:</b> {dataset_name}&nbsp;&nbsp;
   <b>Generated:</b> {result.metadata.get("timestamp", "—")}</p>

<h2>Summary</h2>
<div>
  <div class="stat-box"><div class="stat-num">{n_neurons}</div>
    <div class="stat-lbl">Total Neurons</div></div>
  <div class="stat-box"><div class="stat-num">{n_paired}</div>
    <div class="stat-lbl">Paired</div></div>
  <div class="stat-box"><div class="stat-num">{n_single}</div>
    <div class="stat-lbl">Single-spot</div></div>
  <div class="stat-box"><div class="stat-num">{mean_conf:.2f}</div>
    <div class="stat-lbl">Mean Confidence</div></div>
  <div class="stat-box"><div class="stat-num">{result.metadata.get("n_frames","?")}</div>
    <div class="stat-lbl">Frames</div></div>
  <div class="stat-box"><div class="stat-num">{result.metadata.get("pipeline_elapsed_s","?")}</div>
    <div class="stat-lbl">Elapsed (s)</div></div>
</div>

<h2>Pipeline Metadata</h2>
{_metadata_table_html(result)}

<h2>Detected Neurons</h2>
{_neuron_table_html(result)}

<h2>Summary Maps</h2>
{maps_html}

<h2>Background Mask (valid illumination region)</h2>
{_img_tag(bg_b64, "background_mask", "220px")}

{"<h2>ΔF/F₀ Traces</h2>" + _img_tag(trace_b64, "traces", "100%") if trace_b64 else ""}

{"<h2>Frame Quality</h2>" + _img_tag(fq_b64, "frame_quality", "100%") if fq_b64 else ""}

</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path
