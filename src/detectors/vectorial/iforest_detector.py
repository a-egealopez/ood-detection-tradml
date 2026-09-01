from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from detectors.base import BaseDetector
from detectors.constants import ANOMALY_LABEL, DEFAULT_RANDOM_STATE


class IsolationForestDetector(BaseDetector):
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_samples: float | str = "auto",
        sliced_path: bool | None = None,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.sliced_path = sliced_path
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X_arr = self._to_float(X)
        kwargs: dict[str, Any] = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "max_samples": self.max_samples,
        }
        if self.sliced_path is not None:
            kwargs["sliced_path"] = self.sliced_path

        # sliced_path requires sklearn >= 1.3; degrade gracefully on older versions.
        try:
            self.model = IsolationForest(**kwargs)
        except TypeError:
            self.model = IsolationForest(
                **{k: v for k, v in kwargs.items() if k != "sliced_path"}
            )
        self.model.fit(X_arr)

        raw_train = -self.model.score_samples(X_arr)
        self.score_min = float(raw_train.min())
        self.score_max = float(raw_train.max())
        return self

    def predict(self, X: np.ndarray):
        if self.model is None or self.score_min is None or self.score_max is None:
            raise RuntimeError("You must call fit() before predict()")

        X_arr = self._to_float(X)
        predictions = self.model.predict(X_arr)
        anomalies = self._to_binary_from_labels(predictions, ANOMALY_LABEL)

        raw_scores = -self.model.score_samples(X_arr)
        scores = self._scores_to_unit(raw_scores, self.score_min, self.score_max)

        return anomalies, scores
