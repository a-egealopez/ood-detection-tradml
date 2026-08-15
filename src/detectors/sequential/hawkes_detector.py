"""Hawkes self-exciting point-process detector with an intensity-based score.

Anomaly score from the literature
---------------------------------
The standard way to score anomalies under a Hawkes model is the conditional
(log-)likelihood: fit the self-exciting intensity, then score each observation
by how unlikely its observed event count is given the intensity the model
predicts. Low conditional likelihood => anomalous.

We use a univariate exponential-kernel Hawkes per feature, with the conditional
intensity updated with the exact forward recursion of Ogata (1988):

    lambda_j(t) = mu_j + g_j(t)
    g_j(t+1) = e^{-beta} * (g_j(t) + alpha_j * x_j(t))

where x_j(t) is the observed activity of feature j on day t, mu_j the baseline
intensity, alpha_j the self-excitation strength and beta the exponential decay
(= self.decay). Given lambda_j(t), each day contributes a Poisson log-likelihood
term log(lambda_j(t))*x_j(t) - lambda_j(t). Summing over features and days and
negating gives an anomaly score (higher = more anomalous), normalized to [0, 1].

Backend
-------
``tick`` would be the natural library, but it is a fragile C++ dependency that
on some environments (e.g. this one, tick 0.8.0.2 on Python 3.12) fails to even
construct ``HawkesSumExpKern``. To keep the detector robust and importable, the
intensity recursion is implemented directly with numpy; ``tick`` is no longer a
hard requirement. Each feature is treated as its own node/component.
"""

import numpy as np
from scipy.special import gammaln

from detectors.base import BaseDetector
from detectors.constants import EPSILON, MAX_BRANCHING_RATIO


class HawkesDetector(BaseDetector):
    """Self-exciting point process for event streams (intensity-based score).

    The anomaly score is the negative conditional log-likelihood of each day
    under an exponential-kernel Hawkes model, computed with Ogata's forward
    recursion. A day whose observed activity is unlikely given the predicted
    self-exciting intensity is flagged anomalous.
    """

    def __init__(
        self,
        decay: float = 0.5,
        threshold_percentile: float = 90,
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
        """Negated conditional log-likelihood per row (higher = more anomalous).

        Applies Ogata's recursion forward in time over the whole matrix, then
        returns the per-day negated log-likelihood. Uses only the past, so the
        recursion is causal (no look-ahead).
        """
        X = self._to_float(X)
        n_days, n_features = X.shape

        state = np.zeros(n_features)  # g_j(t): current excitation per feature
        neg_ll = np.zeros(n_days)
        for t in range(n_days):
            intensity = self.baseline_ + state  # λ_j(t) = μ_j + g_j(t)
            intensity = np.clip(intensity, EPSILON, None)

            x = X[t]
            # Full Poisson log-likelihood: x·log(λ) - λ - log(x!). The factorial
            # term is essential: without it, large x would look *more* likely,
            # inverting the anomaly signal (a burst day must be penalized, not
            # rewarded). Higher neg_ll = more anomalous.
            loglik = x @ np.log(intensity) - intensity.sum() - gammaln(x + 1).sum()
            neg_ll[t] = -loglik

            # Ogata forward recursion for the next step.
            state = np.exp(-self.beta_) * (state + self.alpha_ * x)

        return neg_ll

    def fit(self, X: np.ndarray) -> "HawkesDetector":
        X = self._to_float(X)
        self.beta_ = float(self.decay)

        # Baseline mu_j: mean daily activity per feature, floored for stability.
        self.baseline_ = np.clip(X.mean(axis=0), EPSILON, None)

        # Self-excitation alpha_j: a fraction of the baseline. A stable Hawkes
        # process with exponential kernel requires 0 <= alpha < 1 (branching
        # ratio), so we clip to < 1.
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

        X = self._to_float(X)
        scores_raw = self._conditional_loglik(X)

        anomalies = self._above_threshold(scores_raw, self.threshold)
        scores = self._scores_to_unit(scores_raw, self.score_min, self.score_max)

        return anomalies, scores


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_normal = rng.poisson(5, size=(200, 4)).astype(float)
    X_anom = rng.poisson(60, size=(30, 4)).astype(float)  # burst days: 12x activity

    det = HawkesDetector()
    det.fit(X_normal)
    anomalies, scores = det.predict(np.vstack([X_normal, X_anom]))

    n_detected = anomalies[-len(X_anom):].sum()
    print(f" Anomalies detected: {n_detected} / {len(X_anom)}")
    print(f" Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    assert n_detected >= 0.6 * len(X_anom), "should detect high-activity anomalies"
    det._assert_unit_range(scores)
    print(" Validation OK")
