"""Hawkes self-exciting point-process detector (intensity-based score).

Univariate exponential-kernel Hawkes per feature.  Ogata (1988) forward recursion:

    lambda_j(t) = mu_j + g_j(t)
    g_j(t+1)   = e^{-beta} * (g_j(t) + alpha_j * x_j(t))

Score = negated Poisson conditional log-likelihood per day, normalized to [0, 1].

``tick`` would be the natural backend but is a fragile C++ build; the recursion
is implemented directly with numpy instead.
"""

import numpy as np
from scipy.special import gammaln

from detectors.base import BaseDetector
from detectors.constants import (
    DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
    EPSILON,
    MAX_BRANCHING_RATIO,
)


class HawkesDetector(BaseDetector):
    """Self-exciting point process for event streams (intensity-based score)."""

    def __init__(
        self,
        decay: float = 0.5,
        threshold_percentile: float = DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
        alpha_ratio: float = 0.5,
    ):
        self.decay = decay
        self.threshold_percentile = threshold_percentile
        self.alpha_ratio = alpha_ratio
        self.baseline_ = None
        self.alpha_ = None
        self.beta_ = None
        self.threshold = None
        self.score_min = None
        self.score_max = None
        self._fitted = False

    def _conditional_loglik(self, X: np.ndarray) -> np.ndarray:
        """Negated conditional log-likelihood per row (higher = more anomalous)."""
        if self.baseline_ is None or self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("You must call fit() before predict()")
        X = self._to_float(X)
        n_days, n_features = X.shape

        state = np.zeros(n_features)
        neg_ll = np.zeros(n_days)
        for t in range(n_days):
            intensity = np.clip(self.baseline_ + state, EPSILON, None)

            x_t = X[t]
            # x*log(lambda) - lambda - log(x!): factorial term is essential to
            # penalize burst days instead of rewarding them.
            loglik = (
                x_t @ np.log(intensity) - intensity.sum() - gammaln(x_t + 1).sum()
            )
            neg_ll[t] = -loglik

            state = np.exp(-self.beta_) * (state + self.alpha_ * x_t)

        return neg_ll

    def fit(self, X: np.ndarray) -> "HawkesDetector":
        X = self._to_float(X)
        self.beta_ = float(self.decay)

        self.baseline_ = np.clip(X.mean(axis=0), EPSILON, None)

        # alpha_j: fraction of baseline, clipped to < 1 for stability.
        self.alpha_ = np.clip(self.alpha_ratio * self.baseline_, 0.0, MAX_BRANCHING_RATIO)

        scores_train = self._conditional_loglik(X)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = float(np.percentile(scores_train, self.threshold_percentile))
        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            self._check_fitted("_fitted")
        if self.threshold is None or self.score_min is None or self.score_max is None:
            raise RuntimeError("You must call fit() before predict()")

        X = self._to_float(X)
        scores_raw = self._conditional_loglik(X)

        anomalies = self._above_threshold(scores_raw, self.threshold)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores
