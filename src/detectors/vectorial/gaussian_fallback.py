"""Classical (non-robust) Gaussian fallback for the covariance detectors.

The Minimum Covariance Determinant (MCD) robust estimator used by
``EllipticEnvelopeDetector`` and ``RobustCovarianceDetector`` fails on degenerate
data: a zero-variance column or ``n_samples <= n_features`` makes the covariance
singular and sklearn raises instead of fitting. Rather than crash, both detectors
fall back to a classical Gaussian fit (empirical mean / covariance with a small
jitter on the diagonal), which mirrors the defensive style of
``MahalanobisDetector`` and keeps the detector importable on any input.
"""

import numpy as np

from detectors.constants import EPSILON, contamination_percentile


class ClassicalGaussian:
    """Minimal Gaussian model exposing the sklearn interface the detectors use.

    Provides ``location_``, ``covariance_``, ``mahalanobis(X)`` and ``predict(X)``
    so it can transparently replace a fitted ``EllipticEnvelope`` / ``MinCovDet``
    in ``predict`` without changing the detector code paths.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.location_ = None
        self.covariance_ = None
        self._cov_inv = None

    def fit(self, X: np.ndarray) -> "ClassicalGaussian":
        self.location_ = X.mean(axis=0)
        cov = np.cov(X.T)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)
        # Jitter the diagonal so the covariance is always invertible.
        cov = cov + np.eye(cov.shape[0]) * EPSILON
        self.covariance_ = cov
        self._cov_inv = np.linalg.pinv(cov)
        return self

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.location_
        return np.sum((diff @ self._cov_inv) * diff, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self.mahalanobis(X)
        threshold = np.percentile(
            distances, contamination_percentile(self.contamination)
        )
        return np.where(distances > threshold, -1, 1).astype(int)


def is_mcd_safe(X: np.ndarray) -> bool:
    """True when the MCD robust estimator can be fitted on ``X``.

    MCD needs more samples than features and non-constant columns; otherwise
    sklearn raises. This guard lets the covariance detectors decide between the
    robust fit and the classical fallback.
    """
    n_samples, n_features = X.shape
    return n_samples > n_features and (X.std(axis=0) > EPSILON).all()
