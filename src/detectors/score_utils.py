"""Shared score normalization for detectors.

Every detector normalizes its raw anomaly score to [0, 1] (higher = more
anomalous) using the min/max captured during ``fit``. This single helper keeps
the 12 detectors consistent so the ensemble can sum their outputs safely.
"""

import numpy as np

from detectors.constants import EPSILON


def minmax_normalize(
    raw_scores: np.ndarray, score_min: float, score_max: float
) -> np.ndarray:
    """Scale raw scores linearly into [0, 1] using the training min/max."""
    denom = score_max - score_min
    if denom <= 0:
        return np.zeros(np.asarray(raw_scores).shape, dtype=float)
    return np.clip((np.asarray(raw_scores) - score_min) / (denom + EPSILON), 0.0, 1.0)
