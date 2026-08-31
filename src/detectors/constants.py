"""Shared constants and utilities for the detector package."""

import numpy as np

from config import DEFAULT_RANDOM_STATE as DEFAULT_RANDOM_STATE
from config import EPSILON as EPSILON

MAX_SCORE_TOLERANCE = 1e-9
ANOMALY_LABEL = -1
MAX_BRANCHING_RATIO = 0.99
DEFAULT_SUPPORT_FRACTION = 0.7
DEFAULT_DETECTOR_THRESHOLD_PERCENTILE = 95
DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE = 90
DEFAULT_HMM_N_ITER = 100
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_CONTAMINATION = 0.2


def contamination_percentile(contamination: float) -> float:
    """Map contamination fraction to score percentile (top fraction flagged)."""
    return 100.0 * (1.0 - float(contamination))


def minmax_normalize(
    raw_scores: np.ndarray, score_min: float, score_max: float
) -> np.ndarray:
    """Scale raw scores linearly into [0, 1] using the training min/max."""
    denom = score_max - score_min
    if denom <= 0:
        return np.zeros(np.asarray(raw_scores).shape, dtype=float)
    return np.clip((np.asarray(raw_scores) - score_min) / (denom + EPSILON), 0.0, 1.0)
