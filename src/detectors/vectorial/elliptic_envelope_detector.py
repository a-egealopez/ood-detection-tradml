import numpy as np
from sklearn.covariance import EllipticEnvelope

class EllipticEnvelopeDetector:
    def __init__(self, contamination: float = 0.1, robust: bool = True):
        """
        Detecta outliers asumiendo distribución gaussiana.
        robust: usa Minimum Covariance Determinant (MCD) para resistir outliers en fit
        """
        self.contamination = contamination
        self.robust = robust
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "EllipticEnvelopeDetector":
        X = np.asarray(X, dtype=float)
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42,
            support_fraction=None if not self.robust else 0.7
        )
        self.model.fit(X)
        
        scores_train = -self.model.score_samples(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")
        
        X = np.asarray(X, dtype=float)
        predictions = self.model.predict(X)
        anomalies = (predictions == -1).astype(int)
        
        scores_raw = -self.model.mahalanobis(X)
        scores = (scores_raw - self.score_min) / (self.score_max - self.score_min + 1e-8)
        scores = np.clip(scores, 0.0, None)
        
        return anomalies, scores