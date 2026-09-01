import numpy as np
from sklearn.covariance import EllipticEnvelope

from detectors.base import BaseDetector
from detectors.constants import (
    ANOMALY_LABEL,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SUPPORT_FRACTION,
)
from detectors.vectorial.classical_gaussian import ClassicalGaussian, is_mcd_safe


class EllipticEnvelopeDetector(BaseDetector):
    def __init__(self, contamination: float = 0.1, robust: bool = True):
        """Gaussian-ellipsoid boundary; ``robust`` uses the Minimum Covariance
        Determinant (MCD) so outliers present during training do not skew the fit."""
        self.contamination = contamination
        self.robust = robust
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "EllipticEnvelopeDetector":
        X = self._to_float(X)

        if self.robust and is_mcd_safe(X):
            self.model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=DEFAULT_RANDOM_STATE,
                support_fraction=DEFAULT_SUPPORT_FRACTION,
            )
            self.model.fit(X)
        else:
            # Non-robust mode, or degenerate input for MCD (too few samples /
            # constant columns): fall back to a classical Gaussian fit.
            self.model = ClassicalGaussian(contamination=self.contamination)
            self.model.fit(X)

        # Mahalanobis distance grows with anomaly; normalized so higher = anomaly.
        scores_train = self.model.mahalanobis(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        return self

    def predict(self, X: np.ndarray):
        if self.model is None or self.score_min is None or self.score_max is None:
            raise RuntimeError("You must call fit() before predict()")

        X = self._to_float(X)
        predictions = self.model.predict(X)
        anomalies = self._to_binary_from_labels(predictions, ANOMALY_LABEL)

        scores_raw = self.model.mahalanobis(X)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
