import numpy as np
from sklearn.covariance import MinCovDet

from detectors.base import BaseDetector
from detectors.constants import (
    DEFAULT_RANDOM_STATE,
    contamination_percentile,
)
from detectors.vectorial.classical_gaussian import ClassicalGaussian, is_mcd_safe


class RobustCovarianceDetector(BaseDetector):
    def __init__(
        self, contamination: float = 0.1, support_fraction: float | None = None
    ):
        """Minimum Covariance Determinant (MCD): robust mean/covariance estimate.

        More resistant than classical Mahalanobis when outliers are present in the
        training set.
        """
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "RobustCovarianceDetector":
        X = self._to_float(X)

        if is_mcd_safe(X):
            self.model = MinCovDet(
                support_fraction=self.support_fraction,
                random_state=DEFAULT_RANDOM_STATE,
            )
            self.model.fit(X)
        else:
            # Degenerate input (too few samples / constant columns): fall back to a
            # classical Gaussian fit instead of crashing on a singular covariance.
            self.model = ClassicalGaussian(contamination=self.contamination)
            self.model.fit(X)

        scores_train = self.model.mahalanobis(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        # Contamination percentile over the distance distribution sets the threshold.
        self.threshold = float(
            np.percentile(
                scores_train, contamination_percentile(self.contamination)
            )
        )

        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            self._check_fitted("model")

        X = self._to_float(X)

        scores_raw = self.model.mahalanobis(X)

        anomalies = self._above_threshold(scores_raw, self.threshold)

        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
