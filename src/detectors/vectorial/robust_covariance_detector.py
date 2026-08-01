import numpy as np
from sklearn.covariance import MinCovDet


class RobustCovarianceDetector:
    def __init__(
        self, contamination: float = 0.1, support_fraction: float | None = None
    ):
        """
        Minimum Covariance Determinant (MCD): estima media y cov robustas.
        Mejor que Mahalanobis clásico si hay outliers en fit.
        """
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.model = None
        self.mean = None
        self.cov = None
        self.cov_inv = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "RobustCovarianceDetector":
        X = np.asarray(X, dtype=float)

        # Corregido: MinCovDet no acepta 'contamination' en su constructor
        self.model = MinCovDet(support_fraction=self.support_fraction, random_state=42)
        self.model.fit(X)

        self.mean = self.model.location_
        self.cov = self.model.covariance_
        self.cov_inv = self.model.precision_  # Usa la precisión calculada por sklearn

        # Corregido: El método correcto es .mahalanobis(X)
        scores_train = self.model.mahalanobis(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        # El percentil de contaminación se calcula sobre la distribución de las distancias
        self.threshold = float(
            np.percentile(scores_train, 100 * (1 - self.contamination))
        )

        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)

        # Corregido: .mahalanobis(X)
        scores_raw = self.model.mahalanobis(X)

        anomalies = (scores_raw > self.threshold).astype(int)

        # Normalización min-max segura en el rango [0.0, 1.0]
        denom = self.score_max - self.score_min
        if denom == 0:
            scores = np.zeros_like(scores_raw)
        else:
            scores = (scores_raw - self.score_min) / (denom + 1e-8)

        scores = np.clip(scores, 0.0, 1.0)

        return anomalies, scores
