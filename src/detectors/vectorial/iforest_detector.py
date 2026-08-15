import numpy as np
from sklearn.ensemble import IsolationForest

from detectors.base import BaseDetector
from detectors.constants import ANOMALY_LABEL, DEFAULT_RANDOM_STATE


class IsolationForestDetector(BaseDetector):
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_samples: float | str = "auto",
        sliced_path: bool | None = None,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.sliced_path = sliced_path
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X = self._to_float(X)
        kwargs = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "max_samples": self.max_samples,
        }
        if self.sliced_path is not None:
            kwargs["sliced_path"] = self.sliced_path

        # sliced_path requires sklearn >= 1.3; degrade gracefully on older versions.
        try:
            self.model = IsolationForest(**kwargs)
        except TypeError:
            self.model = IsolationForest(
                **{k: v for k, v in kwargs.items() if k != "sliced_path"}
            )
        self.model.fit(X)

        raw_train = -self.model.score_samples(X)
        self.score_min = float(raw_train.min())
        self.score_max = float(raw_train.max())
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            self._check_fitted("model")

        X = self._to_float(X)
        predictions = self.model.predict(X)
        anomalies = self._to_binary_from_labels(predictions, ANOMALY_LABEL)

        raw_scores = -self.model.score_samples(X)
        scores = self._scores_to_unit(raw_scores, self.score_min, self.score_max)

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

    det = IsolationForestDetector(contamination=0.05)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalías detectadas: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    _, single_score = det.predict(X_test[:1])
    det._assert_unit_range(single_score)
    print(" Validación OK")
