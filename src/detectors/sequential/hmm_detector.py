import numpy as np

from detectors.base import BaseDetector
from detectors.constants import DEFAULT_HMM_N_ITER, DEFAULT_RANDOM_STATE


class HMMDetector(BaseDetector):
    """HMM anomaly detector scored with a causal *predictive* log-likelihood.

    The marginal per-observation score ``log p(x_t)`` (used before) is blind to
    temporal context: it treats each day independently, so a day whose value is
    plausible on its own is never flagged, no matter how unlikely its arrival is
    given the preceding days (e.g. an abrupt regime change).

    The predictive score ``log p(x_t | x_{<t})`` conditions on the observed
    history, which is exactly the signal a regime/contextual anomaly produces.
    It is computed as the difference of cumulative sequence log-likelihoods:
    hmmlearn's ``score`` runs the forward filter, so each increment is the
    predictive contribution of the new observation with no look-ahead.
    """

    def __init__(
        self,
        n_components: int = 3,
        threshold_percentile: float = 90,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def _predictive_loglik(self, X: np.ndarray) -> np.ndarray:
        """Causal per-observation predictive log-likelihood of each row."""
        loglik = np.zeros(len(X))
        cumulative = 0.0
        for i in range(len(X)):
            # score(X[:i+1]) - score(X[:i]) = log p(x_{i+1} | x_1..x_i).
            cumulative_next = float(self.model.score(X[: i + 1]))
            loglik[i] = cumulative_next - cumulative
            cumulative = cumulative_next
        return loglik

    def fit(self, X: np.ndarray) -> "HMMDetector":
        # hmmlearn is imported lazily so the core detector package can be imported
        # without it (hmmlearn is an optional/slow dependency; only HMM needs it).
        from hmmlearn import hmm

        X = self._to_float(X)
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            n_iter=DEFAULT_HMM_N_ITER,
            random_state=self.random_state,
        )
        self.model.fit(X)

        scores_train = self._predictive_loglik(X)

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        # The predictive log-likelihood is higher = more normal. Normalize it to
        # [0, 1] with the same canonical (score_min, score_max) pair every detector
        # uses, then invert (1 - x) so that higher = more anomalous, matching the
        # rest of the stack.
        scores_norm = 1.0 - self._scores_to_unit(
            scores_train, self.score_min, self.score_max
        )
        self.threshold = float(np.percentile(scores_norm, self.threshold_percentile))

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            self._check_fitted("model")

        X = self._to_float(X)
        scores = self._predictive_loglik(X)

        # Low predictive log-likelihood = anomaly (inverted normalized score).
        scores_norm = 1.0 - self._scores_to_unit(
            scores, self.score_min, self.score_max
        )

        anomalies = self._above_threshold(scores_norm, self.threshold)

        return anomalies, scores_norm


if __name__ == "__main__":
    # Fase 3 DoD: with a known regime change at time t, the predictive score must
    # spike near t (not be flat), while the old marginal score would stay flat.
    rng = np.random.default_rng(0)
    n_regime_a = 40
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_regime_a, 3))
    X_shift = rng.normal(loc=4.0, scale=1.0, size=(20, 3))  # abrupt regime change
    X = np.vstack([X_normal, X_shift])

    detector = HMMDetector(n_components=2)
    detector.fit(X_normal)
    anomalies, scores = detector.predict(X)

    # The anomaly scores should be concentrated on the shifted segment.
    tail_mean = float(scores[n_regime_a:].mean())
    head_mean = float(scores[:n_regime_a].mean())
    print(f" Mean score head (normal): {head_mean:.3f}")
    print(f" Mean score tail (regime change): {tail_mean:.3f}")
    print(f" Max score index: {int(scores.argmax())} (regime change starts at {n_regime_a})")
    assert tail_mean > head_mean, "predictive HMM did not flag the regime change"
    detector._assert_unit_range(scores)
    print(" Validation OK")
