"""Base class shared by every OOD detector.
    fit(X)
    predict(X)
"""

from abc import ABC, abstractmethod

import numpy as np

from detectors.constants import MAX_SCORE_TOLERANCE, minmax_normalize


def _as_float_array(X) -> np.ndarray:
    """Coerce input to a float numpy array; reject NaN/Inf."""
    float_array = np.asarray(X, dtype=float)
    if not np.isfinite(float_array).all():
        raise ValueError(
            "Input contains NaN or infinite values; detectors require finite data."
        )
    return float_array


class BaseDetector(ABC):
    """Lightweight shared behaviour for all detectors."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit the detector on training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and scores.

        Returns:
            Tuple of (binary_anomalies, scores_in_0_1).
        """
        ...

    def _check_fitted(self, fitted_attr: str) -> None:
        """Raise RuntimeError if the fitted guard attribute is not set."""
        if not getattr(self, fitted_attr, None):
            raise RuntimeError("You must call fit() before predict()")

    def _assert_unit_range(self, scores: np.ndarray) -> None:
        """Sanity-check that normalized scores lie in [0, 1]."""
        if not np.isfinite(scores).all():
            raise ValueError("Scores contain non-finite values")
        if scores.min() < 0.0 or scores.max() > 1.0 + MAX_SCORE_TOLERANCE:
            raise ValueError(
                f"Scores out of [0, 1] range: min={scores.min()}, max={scores.max()}"
            )

    @staticmethod
    def _to_float(X) -> np.ndarray:
        return _as_float_array(X)

    @staticmethod
    def _scores_to_unit(raw_scores, score_min: float, score_max: float) -> np.ndarray:
        return minmax_normalize(raw_scores, score_min, score_max)

    @staticmethod
    def _above_threshold(scores_raw, threshold: float) -> np.ndarray:
        return (scores_raw > threshold).astype(int)

    @staticmethod
    def _to_binary_from_labels(predictions, anomaly_label: int = -1) -> np.ndarray:
        return (predictions == anomaly_label).astype(int)
