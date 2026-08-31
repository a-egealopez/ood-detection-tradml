import numpy as np

from detectors.base import BaseDetector
from detectors.constants import (
    DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
    DEFAULT_HMM_N_ITER,
    DEFAULT_RANDOM_STATE,
)


class HMMDetector(BaseDetector):
    """Predictive log-likelihood anomaly detector (causal, conditions on history)."""

    def __init__(
        self,
        n_components: int = 3,
        threshold_percentile: float = DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
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
        if self.model is None:
            raise RuntimeError("You must call fit() before predict()")
        loglik = np.zeros(len(X))
        cumulative = 0.0
        for i in range(len(X)):
            cumulative_next = float(self.model.score(X[: i + 1]))
            loglik[i] = cumulative_next - cumulative
            cumulative = cumulative_next
        return loglik

    def fit(self, X: np.ndarray) -> "HMMDetector":
        from hmmlearn import hmm  # lazy: only HMM needs hmmlearn

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

        # predictive loglik is higher=more normal; invert so higher=more anomalous
        scores_norm = 1.0 - self._scores_to_unit(
            scores_train, self.score_min, self.score_max
        )
        self.threshold = float(np.percentile(scores_norm, self.threshold_percentile))

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.score_min is None or self.score_max is None or self.threshold is None:
            raise RuntimeError("You must call fit() before predict()")

        X = self._to_float(X)
        scores = self._predictive_loglik(X)

        scores_norm = 1.0 - self._scores_to_unit(
            scores, self.score_min, self.score_max
        )

        anomalies = self._above_threshold(scores_norm, self.threshold)

        return anomalies, scores_norm
