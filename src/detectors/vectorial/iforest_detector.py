import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        max_samples: float | str = "auto",
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X = np.asarray(X, dtype=float)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        self.model.fit(X)

        raw_train = -self.model.score_samples(X)
        self.score_min = float(raw_train.min())
        self.score_max = float(raw_train.max())
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Debes llamar a fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        predictions = self.model.predict(X)
        anomalies = (predictions == -1).astype(int)

        raw_scores = -self.model.score_samples(X)
        scores = (raw_scores - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, 1.0)

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
    assert np.isfinite(single_score).all()
    print(" Validación OK")
