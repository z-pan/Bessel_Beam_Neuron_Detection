#!/usr/bin/env python3
"""Process a single Bessel beam dataset through the segmentation pipeline.

Usage
-----
.. code-block:: bash

    python scripts/run_single.py path/to/data.tif
    python scripts/run_single.py path/to/frames_dir/ --output results/dataset_001/
    python scripts/run_single.py data.tif --config config/default.yaml --output out/

Arguments
---------
data_path
    Path to a multi-frame .tif/.tiff file **or** a directory of single-frame
    .tif files (sorted alphabetically by filename).

--config
    Optional path to a YAML configuration file.  Uses built-in defaults when
    omitted.

--output
    Directory where results are saved.  Created automatically if it does not
    exist.  If omitted, the pipeline runs but no files are written.

--loglevel
    Python logging level: DEBUG, INFO (default), WARNING, ERROR.

Exit codes
----------
0 — success
1 — error (data not found, pipeline failure)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Bessel beam neuron segmentation pipeline on one dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to a multi-frame .tif/.tiff file or a directory of single-frame "
            ".tif files."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (uses built-in defaults if omitted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory for output files (neurons.json, roi_masks.tif, traces.csv, "
            "frame_quality.csv, summary_maps.npz, report.html).  Created if absent."
        ),
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Late import so that --help is always fast
    from bessel_seg.config import PipelineConfig, load_config
    from bessel_seg.pipeline import run_pipeline

    # Load config
    if args.config:
        config = load_config(args.config)
        logger.info("Config loaded from %s", args.config)
    else:
        config = PipelineConfig()
        logger.info("Using default pipeline configuration")

    data_path = Path(args.data_path)
    output_dir = Path(args.output) if args.output else None

    try:
        result = run_pipeline(data_path, config=config, output_dir=output_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    # ------------------------------------------------------------------ #
    # Print human-readable summary to stdout                               #
    # ------------------------------------------------------------------ #
    n = len(result.neurons)
    n_paired = sum(1 for r in result.neurons if r.detection_type == "paired")
    n_single = n - n_paired

    print(f"\n{'=' * 60}")
    print(f"  Dataset : {data_path}")
    if output_dir:
        print(f"  Output  : {output_dir}")
    print(f"  Neurons : {n} total  ({n_paired} paired, {n_single} single-spot)")
    print(f"  Frames  : {result.metadata.get('n_frames', '?')}")
    print(f"  Elapsed : {result.metadata.get('pipeline_elapsed_s', '?')} s")
    print(f"{'=' * 60}\n")

    if result.neurons:
        header = f"  {'ID':>4}  {'Y':>7}  {'X':>7}  {'Type':<20}  {'Conf':>6}  {'Detections':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for roi in result.neurons:
            print(
                f"  {roi.neuron_id:>4}  {roi.center_y:>7.1f}  {roi.center_x:>7.1f}"
                f"  {roi.detection_type:<20}  {roi.confidence:>6.3f}"
                f"  {roi.detection_count:>10}/{roi.total_frames}"
            )
        print()
    else:
        print("  No neurons detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
