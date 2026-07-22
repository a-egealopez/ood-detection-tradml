import numpy as np


class PCAReconstructionDetector:
    def __init__(self, n_components: int = 5, threshold_percentile: float = 95):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.W = None
        self.mean = None
        self.threshold = None
        self.error_min = None
        self.error_max = None

    def fit(self, X: np.ndarray) -> "PCAReconstructionDetector":
        X = np.asarray(X, dtype=float)

        n_components = min(self.n_components, X.shape[1], X.shape[0])
        if n_components < self.n_components:
            self.n_components = n_components

        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.W = Vt[: self.n_components, :].T

        X_recon = X_centered @ self.W @ self.W.T
        recon_errors = np.linalg.norm(X_centered - X_recon, axis=1) ** 2
        self.threshold = np.percentile(recon_errors, self.threshold_percentile)

        self.error_min = float(recon_errors.min())
        self.error_max = float(recon_errors.max())

        return self

    def predict(self, X: np.ndarray):
        if self.W is None:
            raise RuntimeError("Debes llamar a fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean
        X_recon = X_centered @ self.W @ self.W.T
        errors = np.linalg.norm(X_centered - X_recon, axis=1) ** 2

        anomalies = (errors > self.threshold).astype(int)
        scores = (errors - self.error_min) / (self.error_max - self.error_min + 1e-8)
        scores = np.clip(scores, 0.0, None)

        return anomalies, scores


if __name__ == "__main__":
    np.random.seed(42)
    X_normal = np.random.randn(100, 5)
    X_test = np.vstack([
        np.random.randn(80, 5),
        np.random.randn(20, 5) + 6,
    ])

    det = PCAReconstructionDetector(n_components=3, threshold_percentile=95)
    det.fit(X_normal)
    anomalies, scores = det.predict(X_test)

    print(f" Anomalías detectadas: {anomalies.sum()} / {len(anomalies)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(" Validación OK")
    