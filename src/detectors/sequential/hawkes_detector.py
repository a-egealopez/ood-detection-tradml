import numpy as np
from tick.hawkes import HawkesSumExpKern, SimuHawkesMulti
from typing import Tuple

class HawkesDetector:
    def __init__(self, decay: float = 0.5, baseline: float = 0.1, threshold_percentile: float = 90):
        self.decay = decay
        self.baseline = baseline
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "HawkesDetector":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        
        self.model = HawkesSumExpKern([self.decay] * n_features, baseline=[self.baseline] * n_features)
        
        # Convertir X a timestamps por feature
        event_times = []
        for feat in range(n_features):
            times = np.where(X[:, feat] > 0)[0].astype(float)
            event_times.append(times)
        
        self.model.fit(event_times)
        
        # Score = intensidad condicional promedio
        scores_train = np.mean(X, axis=1)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(scores_train, self.threshold_percentile)
        
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")
        
        X = np.asarray(X, dtype=float)
        scores = np.mean(X, axis=1)
        scores_norm = (scores - self.score_min) / (self.score_max - self.score_min + 1e-8)
        scores_norm = np.clip(scores_norm, 0.0, None)
        
        anomalies = (scores > self.threshold).astype(int)
        
        return anomalies, scores_norm