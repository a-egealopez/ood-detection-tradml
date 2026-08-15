"""Shared constants for the detector package.

Centralizes the small numeric constants used across detectors so their purpose
is explicit and they stay consistent (no scattered magic numbers). Tuned once
here, reused everywhere.
"""

import numpy as np

# Numerical stability guard used when dividing by (near-)zero denominators.
EPSILON = 1e-8

# Reproducible pseudo-randomness seed for any detector that fits a stochastic
# model (Isolation Forest, Elliptic Envelope, MinCovDet, ...).
DEFAULT_RANDOM_STATE = 42

# Scores are normalized to [0, 1]; the upper bound is tested with a small
# tolerance because floating-point min-max normalization can overshoot slightly.
MAX_SCORE_TOLERANCE = 1e-9

# Standard anomaly label produced by scikit-learn novelty/outlier detectors.
ANOMALY_LABEL = -1

# Stability constraint for the Hawkes branching ratio: a self-exciting process
# with an exponential kernel is stationary only while alpha < 1.
MAX_BRANCHING_RATIO = 0.99

# Fraction of the training points used as the support of the Minimum Covariance
# Determinant (MCD) robust estimator when the detector runs in "robust" mode.
DEFAULT_SUPPORT_FRACTION = 0.7

# Default cutoff used to turn a continuous score into a binary anomaly flag when
# a detector exposes a `threshold_percentile` parameter (shared across HMM,
# Hawkes, Mahalanobis and PCA Reconstruction).
DEFAULT_THRESHOLD_PERCENTILE = 95

# EM iterations cap for the HMM fit (hmmlearn). Kept here so the magic number is
# not scattered inside the detector file.
DEFAULT_HMM_N_ITER = 100


def contamination_percentile(contamination: float) -> float:
    """Map a contamination fraction to the score percentile above which a point is
    anomalous: the top ``contamination`` fraction of training scores are flagged.

    ``contamination = 0.05`` means "keep the lowest 95% as normal", so the cutoff
    is the 95th percentile (``100 * (1 - 0.05)``). Shared by KNN, RobustCovariance
    and LOF instead of being re-derived in each file.
    """
    return 100.0 * (1.0 - float(contamination))


def as_float_array(X) -> np.ndarray:
    """Coerce input to a float numpy array (the shape detectors expect).

    Also rejects NaN/Inf values up front, so every detector that routes through
    this choke point fails with a clear message instead of silently propagating
    NaNs through means, covariances or SVMs.
    """
    arr = np.asarray(X, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(
            "Input contains NaN or infinite values; detectors require finite data."
        )
    return arr
