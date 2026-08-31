import numpy as np

from features.common import EPSILON


class FeatureScaler:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("You must call fit() before transform()")
        X = np.asarray(X, dtype=float)
        return (X - self.mu) / (self.sigma + EPSILON)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
