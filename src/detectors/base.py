"""Base class shared by every OOD detector.

Each detector implements the same contract (see AGENTS.md):

    fit(X) -> self
    predict(X) -> (anomalies_binary_0_1, anomaly_scores_in_[0, 1])

All of them also share three small pieces of boilerplate: coercing the input to
a float array, guarding ``predict`` against a missing ``fit``, and normalizing
raw scores to [0, 1]. ``BaseDetector`` centralizes those so they are defined
once instead of being copy-pasted across the ~11 detectors.
"""

import numpy as np

from detectors.constants import as_float_array, MAX_SCORE_TOLERANCE
from detectors.score_utils import minmax_normalize


class BaseDetector:
    """Lightweight shared behaviour for all detectors."""

    # Subclasses implement _fit and _predict; they inherit the helpers below.

    @staticmethod
    def _to_float(X) -> np.ndarray:
        """Coerce input to a float numpy array (the shape detectors expect)."""
        return as_float_array(X)

    @staticmethod
    def _check_fitted(fitted_attr: str) -> None:
        """Raise RuntimeError if a detector was used before ``fit``.

        ``fitted_attr`` is the instance attribute (e.g. ``"model"``) that is set
        only after ``fit``. Subclasses pass whichever guard flag they use.
        """
        raise RuntimeError("You must call fit() before predict()")

    @staticmethod
    def _scores_to_unit(raw_scores, score_min: float, score_max: float) -> np.ndarray:
        """Normalize raw scores to [0, 1] using the training min/max."""
        return minmax_normalize(raw_scores, score_min, score_max)

    @staticmethod
    def _above_threshold(scores_raw, threshold: float) -> np.ndarray:
        """Convert continuous scores into a binary 0/1 anomaly flag (0 = normal)."""
        return (scores_raw > threshold).astype(int)

    @staticmethod
    def _to_binary_from_labels(predictions, anomaly_label: int = -1) -> np.ndarray:
        """Map sklearn-style {-1, 1} novelty labels to 0/1 anomaly flags."""
        return (predictions == anomaly_label).astype(int)

    @staticmethod
    def _assert_unit_range(scores: np.ndarray) -> None:
        """Sanity-check that normalized scores lie in [0, 1] (tests/__main__)."""
        assert np.isfinite(scores).all()
        assert scores.min() >= 0.0 and scores.max() <= 1.0 + MAX_SCORE_TOLERANCE
