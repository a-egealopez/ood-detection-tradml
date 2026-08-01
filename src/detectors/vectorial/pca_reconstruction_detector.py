import numpy as np


class PCAReconstructionDetector:
    """PCA reconstruction error detector.

    Fits PCA on the training set, keeps the top `n_components` principal directions,
    and scores a point by its squared reconstruction error (distance to the linear
    subspace). High error = the point does not lie on the dominant manifold = anomaly.

    Notes:
        `n_components` is clamped to `min(n_components, n_features - 1, n_samples - 1)`
        so there is always at least one discarded direction; otherwise the reconstruction
        error would be zero for every point and the detector would be degenerate.
    """

    def __init__(self, n_components: int = 5, threshold_percentile: float = 95):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.W = None
        self.mean = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "PCAReconstructionDetector":
        X = np.asarray(X, dtype=float)
        n_features, n_samples = X.shape[1], X.shape[0]

        effective = min(self.n_components, n_features - 1, n_samples - 1)
        effective = max(effective, 1)
        self.n_components = effective

        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.W = Vt[: self.n_components, :].T

        errors_train = self._reconstruction_errors(X)
        self.threshold = float(np.percentile(errors_train, self.threshold_percentile))
        self.score_min = float(errors_train.min())
        self.score_max = float(errors_train.max())

        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        X_recon = X_centered @ self.W @ self.W.T
        return np.linalg.norm(X_centered - X_recon, axis=1) ** 2

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.W is None or self.mean is None:
            raise RuntimeError("You must call fit() before predict()")

        X = np.asarray(X, dtype=float)
        errors = self._reconstruction_errors(X)

        anomalies = (errors > self.threshold).astype(int)
        scores = (errors - self.score_min) / (self.score_max - self.score_min + 1e-8)
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

    det = PCAReconstructionDetector(n_components=3, threshold_percentile=95)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalies detected: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    assert np.isfinite(scores).all()
    print(" Validation OK")
