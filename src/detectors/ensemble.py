import numpy as np
import pandas as pd

from .zscore_detector import ZScoreDetector
from .isolation_forest_detector import IsolationForestDetector
from .pca_recon_detector import PCAReconstructionDetector


class EnsembleDetector:
    def __init__(self, detectors=None, weights=None, ensemble_threshold_percentile: float = 90):
        if detectors is None:
            detectors = [
                ZScoreDetector(),
                IsolationForestDetector(),
                PCAReconstructionDetector(),
            ]
        self.detectors = detectors
        self.detector_names = [d.__class__.__name__ for d in detectors]

        if weights is None:
            weights = [1.0 / len(detectors)] * len(detectors)
        weights = np.array(weights, dtype=float)
        assert abs(weights.sum() - 1.0) < 1e-6, "Los pesos deben sumar 1"
        assert len(weights) == len(detectors), "weights y detectors deben tener el mismo largo"
        self.weights = weights

        self.ensemble_threshold_percentile = ensemble_threshold_percentile
        self.threshold = None

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        X = np.asarray(X, dtype=float)
        for detector in self.detectors:
            detector.fit(X)

        _, scores_train, _ = self._predict_raw(X)
        self.threshold = np.percentile(scores_train, self.ensemble_threshold_percentile)

        return self

    def _predict_raw(self, X: np.ndarray):
        scores_all = []
        for detector in self.detectors:
            _, score = detector.predict(X)
            scores_all.append(score)

        scores_array = np.array(scores_all)
        score_final = scores_array.T @ self.weights

        return scores_all, score_final, scores_array

    def predict(self, X: np.ndarray):
        if self.threshold is None:
            raise RuntimeError("Debes llamar a fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        scores_all, score_final, _ = self._predict_raw(X)

        anomalies_final = (score_final > self.threshold).astype(int)

        details = pd.DataFrame({
            f"{name.lower()}_score": scores_all[i]
            for i, name in enumerate(self.detector_names)
        })
        details["ensemble_score"] = score_final
        details["is_anomaly"] = anomalies_final
        details["confidence"] = score_final

        return anomalies_final, score_final, details

    def set_weights(self, weights) -> None:
        weights = np.array(weights, dtype=float)
        assert abs(weights.sum() - 1.0) < 1e-6, "Los pesos deben sumar 1"
        assert len(weights) == len(self.detectors)
        self.weights = weights

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold


if __name__ == "__main__":
    np.random.seed(42)
    X_normal = np.random.randn(100, 5)
    X_test = np.vstack([
        np.random.randn(80, 5),
        np.random.randn(20, 5) + 5,
    ])

    ensemble = EnsembleDetector()
    ensemble.fit(X_normal)
    anom, scores, details = ensemble.predict(X_test)

    print(f"  Ensemble: {anom.sum()} anomalías")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Columnas details: {list(details.columns)}")
    print(details.tail(10))

    assert scores[80:].mean() > scores[:80].mean()

    _, single_score, _ = ensemble.predict(X_test[:1])
    assert np.isfinite(single_score).all()

    print(" Validación OK")