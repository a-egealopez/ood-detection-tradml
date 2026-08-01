import numpy as np
from hmmlearn import hmm


class HMMDetector:
    def __init__(self, n_components: int = 3, threshold_percentile: float = 90):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "HMMDetector":
        X = np.asarray(X, dtype=float)
        self.model = hmm.GaussianHMM(n_components=self.n_components, n_iter=100)
        self.model.fit(X)

        self.model.score(X)
        scores_train = np.array([self.model.score(X[i : i + 1]) for i in range(len(X))])

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(scores_train, self.threshold_percentile)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        scores = np.array([self.model.score(X[i : i + 1]) for i in range(len(X))])

        # Log-likelihood bajo = anomalía
        scores_norm = (self.score_max - scores) / (
            self.score_max - self.score_min + 1e-8
        )
        scores_norm = np.clip(scores_norm, 0.0, None)

        anomalies = (scores_norm > (1 - self.threshold / 100)).astype(int)

        return anomalies, scores_norm
