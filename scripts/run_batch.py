#!/usr/bin/env python3
"""Process all Bessel beam datasets in a directory (batch mode).

Usage
-----
.. code-block:: bash

    python scripts/run_batch.py data/raw/
    python scripts/run_batch.py data/raw/ --output data/results/
    python scripts/run_batch.py data/raw/ --config config/default.yaml --jobs 4

Arguments
---------
data_dir
    Root directory containing the datasets.  Each dataset must be either a
    .tif/.tiff file directly inside *data_dir*, or a sub-directory of
    single-frame .tif files.

--config
    Optional path to a YAML configuration file.

--output
    Root output directory.  Each dataset gets a sub-directory named after the
    dataset stem (e.g. ``dataset_001``).  Created automatically.

--jobs  (default: 1)
    Number of parallel worker processes.  Set to 1 to run sequentially.

--loglevel
    Python logging level (default: INFO).

--pattern
    Glob pattern for discovering datasets inside *data_dir*
    (default: ``*`` — all direct children that are .tif files or directories).

Output summary
--------------
After processing, a ``batch_summary.csv`` is written to *output* (if provided)
with columns: dataset, n_neurons, n_paired, n_single, mean_confidence,
pipeline_elapsed_s, status.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-process multiple Bessel beam datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root directory containing datasets (.tif files or sub-directories).",
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
        help="Root output directory for all results.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1, sequential).",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help=(
            "Glob pattern relative to data_dir for dataset discovery "
            "(default: '*' — all direct children)."
        ),
    )
    return parser


def _discover_datasets(data_dir: Path, pattern: str) -> list[Path]:
    """Return sorted list of dataset paths matching *pattern*.

    A valid dataset is either:
    * A .tif / .tiff file.
    * A directory containing at least one .tif / .tiff file.
    """
    candidates = sorted(data_dir.glob(pattern))
    datasets: list[Path] = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            datasets.append(p)
        elif p.is_dir() and (list(p.glob("*.tif")) or list(p.glob("*.tiff"))):
            datasets.append(p)
    return datasets


def _process_one(
    dataset_path: Path,
    config,
    output_root: Path | None,
) -> dict:
    """Process a single dataset; return a summary dict."""
    import numpy as np
    from bessel_seg.pipeline import run_pipeline

    output_dir = (output_root / dataset_path.stem) if output_root else None

    try:
        result = run_pipeline(dataset_path, config=config, output_dir=output_dir)
        neurons = result.neurons
        n = len(neurons)
        n_paired = sum(1 for r in neurons if r.detection_type == "paired")
        mean_conf = float(np.mean([r.confidence for r in neurons])) if neurons else 0.0
        return {
            "dataset": str(dataset_path.name),
            "n_neurons": n,
            "n_paired": n_paired,
            "n_single": n - n_paired,
            "mean_confidence": round(mean_conf, 4),
            "pipeline_elapsed_s": result.metadata.get("pipeline_elapsed_s", ""),
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).error(
            "Failed to process %s: %s", dataset_path, exc
        )
        traceback.print_exc()
        return {
            "dataset": str(dataset_path.name),
            "n_neurons": 0,
            "n_paired": 0,
            "n_single": 0,
            "mean_confidence": 0.0,
            "pipeline_elapsed_s": "",
            "status": f"error: {exc}",
        }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    import pandas as pd
    from bessel_seg.config import PipelineConfig, load_config

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("data_dir does not exist or is not a directory: %s", data_dir)
        return 1

    output_root = Path(args.output) if args.output else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config) if args.config else PipelineConfig()
    if args.config:
        logger.info("Config loaded from %s", args.config)

    datasets = _discover_datasets(data_dir, args.pattern)
    if not datasets:
        logger.error(
            "No datasets found in %s matching pattern '%s'", data_dir, args.pattern
        )
        return 1

    logger.info("Found %d datasets in %s", len(datasets), data_dir)

    # ------------------------------------------------------------------ #
    # Process datasets: sequential or parallel                             #
    # ------------------------------------------------------------------ #
    summaries: list[dict] = []

    if args.jobs == 1:
        # Sequential — simplest and easiest to debug
        for dataset_path in datasets:
            logger.info("Processing: %s", dataset_path.name)
            summary = _process_one(dataset_path, config, output_root)
            summaries.append(summary)
            status = summary["status"]
            if status == "ok":
                logger.info(
                    "  ✓ %s — %d neurons, %.1fs",
                    dataset_path.name,
                    summary["n_neurons"],
                    float(summary["pipeline_elapsed_s"] or 0),
                )
            else:
                logger.warning("  ✗ %s — %s", dataset_path.name, status)
    else:
        import multiprocessing
        from functools import partial

        logger.info("Launching %d worker processes", args.jobs)
        worker = partial(_process_one, config=config, output_root=output_root)
        with multiprocessing.Pool(processes=args.jobs) as pool:
            summaries = pool.map(worker, datasets)

    # ------------------------------------------------------------------ #
    # Print and save batch summary                                         #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(summaries)
    print("\n" + df.to_string(index=False) + "\n")

    n_ok = (df["status"] == "ok").sum()
    n_fail = len(df) - n_ok
    print(f"Batch complete: {n_ok}/{len(datasets)} succeeded, {n_fail} failed.\n")

    if output_root:
        summary_path = output_root / "batch_summary.csv"
        df.to_csv(summary_path, index=False)
        logger.info("Batch summary written to %s", summary_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
