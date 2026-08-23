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
            intensity = self.baseline_ + state  # lambda_j(t) = mu_j + g_j(t)
            intensity = np.clip(intensity, EPSILON, None)

            x_t = X[t]
            # Full Poisson log-likelihood: x*log(lambda) - lambda - log(x!). The factorial
            # term is essential: without it, large x would look *more* likely,
            # inverting the anomaly signal (a burst day must be penalized, not
            # rewarded). Higher neg_ll = more anomalous.
            loglik = (
                x_t @ np.log(intensity) - intensity.sum() - gammaln(x_t + 1).sum()
            )
            neg_ll[t] = -loglik

            # Ogata forward recursion for the next step.
            state = np.exp(-self.beta_) * (state + self.alpha_ * x_t)

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

    # DoD test 1: a burst of correlated events must spike the score at its onset,
    # and low-activity stretches must stay quiet. Hourly-count view (24 components),
    # as the pipeline uses. Note: a self-exciting model adapts to a *sustained*
    # burst after the first day (bursts beget bursts), so the spike marks the onset.
    n_normal = 60
    hourly = rng.poisson(5, size=(n_normal, 24)).astype(float)
    burst = rng.poisson(40, size=(2, 24)).astype(float)  # 8x activity days
    tail = rng.poisson(5, size=(20, 24)).astype(float)
    X = np.vstack([hourly, burst, tail])

    detector = HawkesDetector()
    detector.fit(hourly)
    anomalies, scores = detector.predict(X)

    burst_window = slice(n_normal, n_normal + len(burst))
    quiet_window = slice(n_normal + len(burst), len(X))
    print(f" Burst-window max score: {scores[burst_window].max():.3f}")
    print(f" Quiet-tail mean score:  {scores[quiet_window].mean():.3f}")
    assert scores[burst_window].max() >= scores[quiet_window].max(), (
        "Hawkes should spike at the burst onset, not in quiet stretches"
    )
    detector._assert_unit_range(scores)

    # DoD test 2: a temporal *displacement* (events moved from a busy hour into a
    # normally quiet night hour, totals unchanged) must also raise the score.
    normal_day = rng.poisson(5, size=(24,)).astype(float)
    displaced_day = normal_day.copy()
    moved = min(6, int(displaced_day[12]))  # move some midday events to 03:00
    displaced_day[12] -= moved
    displaced_day[3] += moved

    hourly2 = rng.poisson(5, size=(60, 24)).astype(float)
    X2 = np.vstack([hourly2, displaced_day[None, :], hourly2[:3]])
    detector2 = HawkesDetector()
    detector2.fit(hourly2)
    _, scores2 = detector2.predict(X2)

    displaced_score = float(scores2[60])
    baseline_score = float(scores2[61:].mean())
    print(f" Displaced-day score: {displaced_score:.3f} vs baseline {baseline_score:.3f}")
    assert displaced_score > baseline_score, "Hawkes should flag a temporal displacement"
    detector2._assert_unit_range(scores2)

    print(" Validation OK")
