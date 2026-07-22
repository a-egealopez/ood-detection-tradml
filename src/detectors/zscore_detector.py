import numpy as np


class ZScoreDetector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self

    def predict(self, X: np.ndarray):
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Debes llamar a fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        z = (X - self.mu) / (self.sigma + 1e-8)

        anomaly_per_feature = np.abs(z) > self.threshold
        anomalies = anomaly_per_feature.any(axis=1).astype(int)
        scores = np.max(np.abs(z), axis=1)

        return anomalies, scores


if __name__ == "__main__":
    np.random.seed(42)
    X_normal = np.random.randn(100, 5)
    X_test = np.vstack([
        np.random.randn(80, 5),
        np.random.randn(20, 5) + 6,
    ])

    det = ZScoreDetector(threshold=3.0)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalías detectadas: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    assert anomalies[80:].sum() > 10
    print(" Validación OK")