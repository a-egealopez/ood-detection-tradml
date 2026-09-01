"""First-order Markov transition-likelihood detector.

Wraps ``NextEventTransitionExtractor``: learns normal transition probabilities
from training events, reduces each day to
``[mean_logprob, min_logprob, rare_transition_rate]``.
Anomalous days have low transition likelihood under the learned model.
"""

import numpy as np

from detectors.base import BaseDetector
from detectors.constants import DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE
from features import NextEventTransitionExtractor

MEAN_LOGPROB_COLUMN = 0


class MarkovSequenceDetector(BaseDetector):
    """Sequence detector scoring days by the likelihood of their transitions."""

    def __init__(self, threshold_percentile: float = DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE):
        self.threshold_percentile = threshold_percentile
        self.extractor = NextEventTransitionExtractor()
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit_extractor(self, df_train) -> "MarkovSequenceDetector":
        """Learn normal transition probabilities from training events.

        Must be called before ``fit``; ``fit`` receives the extracted feature matrix.
        """
        self.extractor.fit(df_train)
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # Negate so higher = more anomalous.
        return -np.asarray(X, dtype=float)[:, MEAN_LOGPROB_COLUMN]

    def fit(self, X: np.ndarray) -> "MarkovSequenceDetector":
        X = self._to_float(X)
        raw = self._raw_scores(X)

        self.score_min = float(raw.min())
        self.score_max = float(raw.max())

        scores_norm = self._scores_to_unit(raw, self.score_min, self.score_max)
        self.threshold = float(np.percentile(scores_norm, self.threshold_percentile))

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.score_min is None
            or self.score_max is None
            or self.threshold is None
        ):
            raise RuntimeError("You must call fit() before predict()")

        X = self._to_float(X)
        raw = self._raw_scores(X)
        scores = self._scores_to_unit(raw, self.score_min, self.score_max)

        anomalies = self._above_threshold(scores, self.threshold)

        return anomalies, scores
