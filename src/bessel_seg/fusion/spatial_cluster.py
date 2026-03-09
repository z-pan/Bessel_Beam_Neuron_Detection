"""Spatial clustering of SpotPair detections across frames.

Neurons are spatially stable across the recording.  A single neuron will
produce multiple :class:`~bessel_seg.data_types.SpotPair` detections — one
per active frame — all centred within a few pixels of its true position.

This module clusters those per-frame detections by the (y, x) centre of each
SpotPair using DBSCAN, producing one cluster per candidate neuron.  Frame
indices are intentionally ignored: the same neuron in frames 1 and 150 must
end up in the same cluster.

Calibrated parameters (from CLAUDE.md):
  - ``eps = 8.0`` px: same neuron across frames should have centres within a
    few pixels; 8 px is generous to accommodate sub-pixel localisation error.
  - ``min_samples = 3``: require at least 3 detections to form a cluster,
    consistent with the expected 3–7 frame event duration and at least a few
    active events per recording.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from bessel_seg.config import ClusteringConfig
from bessel_seg.data_types import SpotPair

logger = logging.getLogger(__name__)


def cluster_neuron_detections(
    all_pairs: list[SpotPair],
    config: ClusteringConfig,
) -> list[list[SpotPair]]:
    """Cluster SpotPair detections across frames by spatial proximity.

    Uses DBSCAN on the (y, x) centre coordinates of each SpotPair.  Frame
    index is deliberately excluded so that the same neuron firing in different
    frames is always grouped together.

    Noise points (DBSCAN label ``-1``) are silently discarded.

    Args:
        all_pairs: All :class:`~bessel_seg.data_types.SpotPair` objects from
            all frames combined (paired detections + rescued orphan pairs).
        config: :class:`~bessel_seg.config.ClusteringConfig` carrying
            ``eps`` and ``min_samples``.

    Returns:
        List of clusters.  Each cluster is a non-empty list of SpotPair
        objects that belong to the same candidate neuron.  The list is
        ordered by decreasing cluster size.
    """
    if not all_pairs:
        logger.info("cluster_neuron_detections: no pairs to cluster.")
        return []

    # Extract (y, x) centres for DBSCAN
    centres: NDArray[np.float64] = np.array(
        [p.center for p in all_pairs], dtype=np.float64
    )  # (N, 2)

    db = DBSCAN(eps=config.eps, min_samples=config.min_samples, metric="euclidean")
    labels: NDArray[np.int64] = db.fit_predict(centres)

    # Group pairs by cluster label (skip noise = -1)
    cluster_map: dict[int, list[SpotPair]] = {}
    for pair, label in zip(all_pairs, labels):
        if label == -1:
            continue
        cluster_map.setdefault(int(label), []).append(pair)

    # Sort clusters by descending size (larger clusters → more evidence)
    clusters = sorted(cluster_map.values(), key=len, reverse=True)

    n_noise = int((labels == -1).sum())
    logger.info(
        "cluster_neuron_detections: %d pairs → %d clusters, %d noise points "
        "(eps=%.1f, min_samples=%d)",
        len(all_pairs),
        len(clusters),
        n_noise,
        config.eps,
        config.min_samples,
    )
    return clusters
