import numpy as np


class ZScoreDetector:
    """Univariate Z-score detector (per-feature) with score normalization to [0, 1].

    A point is anomalous when any of its per-feature absolute z-scores exceeds the
    threshold (in standard deviations from the training mean). The returned score is
    the max |z| over features, normalized to [0, 1] using the training distribution.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu = None
        self.sigma = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)

        zmax_train = self._max_abs_z(X)
        self.score_min = float(zmax_train.min())
        self.score_max = float(zmax_train.max())
        return self

    def _max_abs_z(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mu) / (self.sigma + 1e-8)
        return np.max(np.abs(z), axis=1)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("You must call fit() before predict()")

        X = np.asarray(X, dtype=float)
        zmax = self._max_abs_z(X)

        anomalies = (zmax > self.threshold).astype(int)
        scores = (zmax - self.score_min) / (self.score_max - self.score_min + 1e-8)
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

    det = ZScoreDetector(threshold=3.0)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalies detected: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    assert anomalies[80:].sum() > 10
    assert np.isfinite(scores).all()
    print(" Validation OK")
