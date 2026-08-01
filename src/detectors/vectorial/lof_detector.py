import numpy as np
from pyod.models.lof import LOF


class LOFDetector:
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        X = np.asarray(X, dtype=float)
        self.model = LOF(n_neighbors=self.n_neighbors, contamination=self.contamination)
        self.model.fit(X)

        scores_train = self.model.decision_scores_
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        scores_raw = self.model.decision_function(X)
        anomalies = self.model.predict(X).astype(int)

        scores = (scores_raw - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, None)

        return anomalies, scores
