from typing import Literal

import numpy as np
import pandas as pd

from detectors.constants import EPSILON


class EnsembleDetector:
    def __init__(
        self,
        detectors: list | None = None,
        weights: np.ndarray = None,
        ensemble_mode: Literal["soft", "hard"] = "soft",
        ensemble_threshold_percentile: float = 90,
    ):
        """Combine per-detector scores into one anomaly verdict.

        ensemble_mode:
            - "soft": Weighted Sum Rule (scores in [0,1] -> weighted sum)
            - "hard": Majority Voting Rule - each detector votes 1/0 with its own
              learned threshold; a day is flagged when a weighted majority votes
              anomaly (score_final > 0.5).

        weights: per-detector weight vector. If None, uniform.
        """
        if detectors is None:
            detectors = []

        self.detectors = detectors
        self.detector_names = [d.__class__.__name__ for d in detectors]
        self.ensemble_mode = ensemble_mode
        self.ensemble_threshold_percentile = ensemble_threshold_percentile
        self.threshold = None

        if weights is None:
            weights = (
                np.ones(len(detectors)) / len(detectors) if detectors else np.array([])
            )
        else:
            weights = np.asarray(weights, dtype=float)

        if len(weights) > 0:
            assert abs(weights.sum() - 1.0) < EPSILON, (
                f"Weights must sum to 1 (got {weights.sum()})"
            )
        assert len(weights) == len(detectors), "Len(weights) != Len(detectors)"
        self.weights = weights

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        """Fit every detector and compute the ensemble threshold."""
        X = np.asarray(X, dtype=float)
        for detector in self.detectors:
            detector.fit(X)

        _, scores_train = self._predict_raw(X)

        if self.ensemble_mode == "hard":
            # Majority rule is fixed: a point is anomalous when a weighted
            # majority of detectors vote yes (score_final > 0.5). A percentile
            # over the discrete vote-sum would land on the maximum (1.0) and
            # flag nothing, so we do not use it for hard voting.
            self.threshold = 0.5
        else:
            self.threshold = np.percentile(
                scores_train, self.ensemble_threshold_percentile
            )

        return self

    def _predict_raw(self, X: np.ndarray):
        """Per-detector scores and the combined score (before the threshold)."""
        scores_per_detector = []
        votes_per_detector = []
        for detector in self.detectors:
            # Each detector's binary verdict uses its OWN learned threshold
            # (captured during fit); the continuous score is kept for the
            # soft mode and for the details table.
            anomalies, score = detector.predict(X)
            scores_per_detector.append(score)
            votes_per_detector.append(anomalies.astype(float))

        scores_array = np.array(scores_per_detector)  # shape: (n_detectors, n_samples)

        if self.ensemble_mode == "soft":
            # Sum rule: S_ensemble(x) = sum_i w_i * s_i(x)
            score_final = scores_array.T @ self.weights
        elif self.ensemble_mode == "hard":
            # Majority Voting: each detector contributes its binary vote
            # (0/1) and the ensemble score is the weighted vote share in [0,1].
            votes = np.array(votes_per_detector)  # shape: (n_detectors, n_samples)
            score_final = votes.T @ self.weights
        else:
            raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")

        return scores_per_detector, score_final

    def predict(self, X: np.ndarray):
        """Predict with the ensemble threshold."""
        if self.threshold is None:
            raise RuntimeError("You must call fit() before predict()")

        X = np.asarray(X, dtype=float)
        scores_all, score_final = self._predict_raw(X)

        anomalies_final = (score_final > self.threshold).astype(int)

        details = pd.DataFrame(
            {
                f"{name.lower()}_score": scores_all[i]
                for i, name in enumerate(self.detector_names)
            }
        )
        details["ensemble_score"] = score_final
        details["is_anomaly"] = anomalies_final
        details["confidence"] = score_final

        return anomalies_final, score_final, details
