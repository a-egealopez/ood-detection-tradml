import numpy as np
from sklearn.svm import OneClassSVM

from detectors.base import BaseDetector
from detectors.constants import ANOMALY_LABEL


class OCSVMDetector(BaseDetector):
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "auto"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "OCSVMDetector":
        X = self._to_float(X)
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X)

        # decision_function is positive for inliers -> negate so that higher = anomaly.
        scores_train = -self.model.decision_function(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            self._check_fitted("model")

        X = self._to_float(X)
        predictions = self.model.predict(X)
        anomalies = self._to_binary_from_labels(predictions, ANOMALY_LABEL)

        scores_raw = -self.model.decision_function(X)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
