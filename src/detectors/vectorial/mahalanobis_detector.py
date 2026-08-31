import numpy as np
from scipy.spatial.distance import mahalanobis

from detectors.base import BaseDetector
from detectors.constants import DEFAULT_DETECTOR_THRESHOLD_PERCENTILE, EPSILON


class MahalanobisDetector(BaseDetector):
    def __init__(self, threshold_percentile: float = DEFAULT_DETECTOR_THRESHOLD_PERCENTILE):
        self.threshold_percentile = threshold_percentile
        self.mean = None
        self.cov = None
        self.cov_inv = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        X = self._to_float(X)
        self.mean = X.mean(axis=0)
        self.cov = np.cov(X.T)

        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            # Singular covariance: add a small diagonal jitter to make it invertible.
            self.cov += np.eye(self.cov.shape[0]) * EPSILON
            self.cov_inv = np.linalg.inv(self.cov)

        scores_train = np.array(
            [mahalanobis(X[i], self.mean, self.cov_inv) for i in range(len(X))]
        )

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(scores_train, self.threshold_percentile)

        return self

    def predict(self, X: np.ndarray):
        if self.mean is None or self.cov_inv is None:
            self._check_fitted("mean/cov_inv")

        X = self._to_float(X)
        scores_raw = np.array(
            [mahalanobis(X[i], self.mean, self.cov_inv) for i in range(len(X))]
        )

        anomalies = self._above_threshold(scores_raw, self.threshold)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
