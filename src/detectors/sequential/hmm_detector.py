import numpy as np

from detectors.base import BaseDetector
from detectors.constants import DEFAULT_HMM_N_ITER


class HMMDetector(BaseDetector):
    def __init__(
        self,
        n_components: int = 3,
        threshold_percentile: float = 90,
    ):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "HMMDetector":
        # hmmlearn is imported lazily so the core detector package can be imported
        # without it (hmmlearn is an optional/slow dependency; only HMM needs it).
        from hmmlearn import hmm

        X = self._to_float(X)
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, n_iter=DEFAULT_HMM_N_ITER
        )
        self.model.fit(X)

        scores_train = np.array([self.model.score(X[i : i + 1]) for i in range(len(X))])

        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())

        # The HMM score is a log-likelihood: higher = more normal. Normalize it to
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
        scores = np.array([self.model.score(X[i : i + 1]) for i in range(len(X))])

        # Low log-likelihood = anomaly (inverted normalized score).
        scores_norm = 1.0 - self._scores_to_unit(
            scores, self.score_min, self.score_max
        )

        anomalies = self._above_threshold(scores_norm, self.threshold)

        return anomalies, scores_norm
