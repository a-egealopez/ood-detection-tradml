import numpy as np
from scipy.spatial.distance import mahalanobis


class MahalanobisDetector:
    def __init__(self, threshold_percentile: float = 95):
        self.threshold_percentile = threshold_percentile
        self.mean = None
        self.cov = None
        self.cov_inv = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.cov = np.cov(X.T)

        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            # Singular: regularizar
            self.cov += np.eye(self.cov.shape[0]) * 1e-6
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
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        scores_raw = np.array(
            [mahalanobis(X[i], self.mean, self.cov_inv) for i in range(len(X))]
        )

        anomalies = (scores_raw > self.threshold).astype(int)
        scores = (scores_raw - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, None)

        return anomalies, scores
