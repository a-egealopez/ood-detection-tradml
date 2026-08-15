import numpy as np
from sklearn.neighbors import NearestNeighbors

from detectors.base import BaseDetector
from detectors.constants import contamination_percentile


class KNNDetector(BaseDetector):
    def __init__(self, n_neighbors: int = 5, contamination: float = 0.1):
        """Outlier = distance to the k-th nearest training neighbor is very high.

        Simple, robust, and makes no distributional assumption.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "KNNDetector":
        X = self._to_float(X)
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.model.fit(X)

        distances, _ = self.model.kneighbors(X)
        scores_train = distances[:, -1]  # Distancia al k-ésimo vecino

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(
            scores_train, contamination_percentile(self.contamination)
        )

        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            self._check_fitted("model")

        X = self._to_float(X)
        distances, _ = self.model.kneighbors(X)
        scores_raw = distances[:, -1]

        anomalies = self._above_threshold(scores_raw, self.threshold)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
