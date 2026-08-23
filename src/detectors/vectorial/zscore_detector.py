import numpy as np

from detectors.base import BaseDetector
from detectors.constants import EPSILON


class ZScoreDetector(BaseDetector):
    """Univariate Z-score detector (per-feature) with score normalization to [0, 1].

    A point is anomalous when any of its per-feature absolute z-scores exceeds the
    threshold (in standard deviations from the training mean). The returned score is
    the max |z| over features, normalized to [0, 1] using the training distribution.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu = None
        self.sigma = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        X_arr = self._to_float(X)
        self.mu = X_arr.mean(axis=0)
        self.sigma = X_arr.std(axis=0)

        zmax_train = self._max_abs_z(X_arr)
        self.score_min = float(zmax_train.min())
        self.score_max = float(zmax_train.max())
        return self

    def _max_abs_z(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mu) / (self.sigma + EPSILON)
        return np.max(np.abs(z), axis=1)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mu is None or self.sigma is None:
            self._check_fitted("mu/sigma")

        X_arr = self._to_float(X)
        zmax = self._max_abs_z(X_arr)

        anomalies = self._above_threshold(zmax, self.threshold)
        scores = self._scores_to_unit(zmax, self.score_min, self.score_max)

        return anomalies, scores


if __name__ == "__main__":
    np.random.seed(42)
    X_normal = np.random.randn(100, 5)
    X_test = np.vstack(
        [
            np.random.randn(80, 5),
            np.random.randn(20, 5) + 6,
        ]
    )

    det = ZScoreDetector(threshold=3.0)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalies detected: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    assert anomalies[80:].sum() > 10
    det._assert_unit_range(scores)
    print(" Validation OK")
