import numpy as np

from detectors.base import BaseDetector


class LOFDetector(BaseDetector):
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        # pyod is imported lazily so the core detector package can be imported
        # without it (pyod is an optional dependency; only LOF uses it here).
        from pyod.models.lof import LOF

        X = self._to_float(X)
        self.model = LOF(n_neighbors=self.n_neighbors, contamination=self.contamination)
        self.model.fit(X)

        scores_train = np.asarray(self.model.decision_scores_)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.score_min is None or self.score_max is None:
            raise RuntimeError("You must call fit() before predict()")

        X = self._to_float(X)
        scores_raw = np.asarray(self.model.decision_function(X))
        anomalies = np.asarray(self.model.predict(X)).astype(int)

        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
