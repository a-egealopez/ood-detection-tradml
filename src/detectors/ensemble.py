import numpy as np
import pandas as pd
from typing import List, Literal

class EnsembleDetector:
    def __init__(
        self,
        detectors: List = None,
        weights: np.ndarray = None,
        ensemble_mode: Literal["soft", "hard"] = "soft",
        ensemble_threshold_percentile: float = 90,
    ):
        """
        Ensemble Anomaly Detection.
        
        ensemble_mode:
            - "soft": Weighted Sum Rule (scores ∈ [0,1] → suma ponderada)
            - "hard": Majority Voting Rule (binario, votación mayoritaria)
        
        weights: vector ponderación detectores. Si None, uniforme.
        """
        if detectors is None:
            detectors = []
        
        self.detectors = detectors
        self.detector_names = [d.__class__.__name__ for d in detectors]
        self.ensemble_mode = ensemble_mode
        self.ensemble_threshold_percentile = ensemble_threshold_percentile
        self.threshold = None
        
        if weights is None:
            weights = np.ones(len(detectors)) / len(detectors) if detectors else np.array([])
        else:
            weights = np.asarray(weights, dtype=float)
        
        if len(weights) > 0:
            assert abs(weights.sum() - 1.0) < 1e-6, f"Pesos no suman 1, suman {weights.sum()}"
        assert len(weights) == len(detectors), "Len(weights) != Len(detectors)"
        self.weights = weights

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        """Entrenar todos detectores + threshold en ensemble."""
        X = np.asarray(X, dtype=float)
        for detector in self.detectors:
            detector.fit(X)
        
        _, scores_train = self._predict_raw(X)
        self.threshold = np.percentile(scores_train, self.ensemble_threshold_percentile)
        
        return self

    def _predict_raw(self, X: np.ndarray):
        """Calcular scores raw sin aplicar threshold."""
        scores_all = []
        for detector in self.detectors:
            _, score = detector.predict(X)
            scores_all.append(score)
        
        scores_array = np.array(scores_all)  # shape: (n_detectors, n_samples)
        
        if self.ensemble_mode == "soft":
            # Sum Rule: S_ensemble(x) = Σ w_i * s_i(x)
            score_final = scores_array.T @ self.weights
        elif self.ensemble_mode == "hard":
            # Majority Voting: cada detector vota 0/1 si score > mediana
            threshs = np.array([np.median(s) for s in scores_all])
            votes = np.array([
                (scores_all[i] > threshs[i]).astype(float)
                for i in range(len(self.detectors))
            ])
            # Score ensemble = weighted sum of votes (0 a 1)
            score_final = votes.T @ self.weights
        else:
            raise ValueError(f"Modo desconocido: {self.ensemble_mode}")
        
        return scores_all, score_final

    def predict(self, X: np.ndarray):
        """Predicción con threshold."""
        if self.threshold is None:
            raise RuntimeError("Llama fit() antes de predict()")
        
        X = np.asarray(X, dtype=float)
        scores_all, score_final = self._predict_raw(X)
        
        anomalies_final = (score_final > self.threshold).astype(int)
        
        details = pd.DataFrame({
            f"{name.lower()}_score": scores_all[i]
            for i, name in enumerate(self.detector_names)
        })
        details["ensemble_score"] = score_final
        details["is_anomaly"] = anomalies_final
        details["confidence"] = score_final
        
        return anomalies_final, score_final, details

    def set_weights(self, weights: np.ndarray) -> None:
        """Cambiar pesos post-entrenamiento."""
        weights = np.array(weights, dtype=float)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert len(weights) == len(self.detectors)
        self.weights = weights

    def set_threshold(self, threshold: float) -> None:
        """Cambiar threshold post-entrenamiento."""
        self.threshold = threshold

    def get_config(self) -> dict:
        """Retorna config actual (para debugging/logging)."""
        return {
            "n_detectors": len(self.detectors),
            "detector_names": self.detector_names,
            "weights": self.weights.tolist(),
            "ensemble_mode": self.ensemble_mode,
            "threshold": self.threshold,
            "ensemble_threshold_percentile": self.ensemble_threshold_percentile,
        }