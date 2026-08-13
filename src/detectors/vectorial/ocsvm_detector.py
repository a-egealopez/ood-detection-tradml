import numpy as np
from sklearn.svm import OneClassSVM


class OCSVMDetector:
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "auto"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "OCSVMDetector":
        X = np.asarray(X, dtype=float)
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X)

        # decision_function is positive for inliers -> negate so that higher = anomaly.
        scores_train = -self.model.decision_function(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        predictions = self.model.predict(X)
        anomalies = (predictions == -1).astype(int)

        scores_raw = -self.model.decision_function(X)
        scores = (scores_raw - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, 1.0)

        return anomalies, scores
