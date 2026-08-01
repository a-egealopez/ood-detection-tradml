import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNDetector:
    def __init__(self, n_neighbors: int = 5, contamination: float = 0.1):
        """
        Outlier = distancia a k-ésimo vecino más lejano en X_train es muy alta.
        Muy simple, muy robusto, no asume forma de distribución.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "KNNDetector":
        X = np.asarray(X, dtype=float)
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.model.fit(X)

        distances, _ = self.model.kneighbors(X)
        scores_train = distances[:, -1]  # Distancia al k-ésimo vecino

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(scores_train, 100 * (1 - self.contamination))

        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        distances, _ = self.model.kneighbors(X)
        scores_raw = distances[:, -1]

        anomalies = (scores_raw > self.threshold).astype(int)
        scores = (scores_raw - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, None)

        return anomalies, scores
