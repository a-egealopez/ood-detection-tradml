"""MarkovSequenceDetector: first-order Markov transition-likelihood detector.

Wraps ``NextEventTransitionExtractor`` (the DeepLog-style next-event model). The
extractor learns the normal transition probabilities from the training events and
reduces each day to per-day features:

    [mean_logprob, min_logprob, rare_transition_rate]

A day whose sensor transitions are mostly improbable under the learned model is
anomalous: this is the collective/order anomaly family (the joint *order* of the
events deviates, while the marginal counts stay normal).

The extractor is fitted on the training events via ``fit_extractor(df_train)``;
``fit``/``predict`` then follow the standard detector contract on the extracted
per-day feature matrix (a proper ndarray, so it flows through the ensemble's
``detector_inputs``). The anomaly score is the mean negative log-probability of
the day's transitions, normalized to [0, 1] against the training distribution.
"""

import numpy as np

from detectors.base import BaseDetector
from features import NextEventTransitionExtractor

# Column of the extractor's per-day feature matrix holding the mean transition
# log-probability (negative values, higher = more likely = more normal).
MEAN_LOGPROB_COLUMN = 0


class MarkovSequenceDetector(BaseDetector):
    """Sequence detector scoring days by the likelihood of their transitions."""

    def __init__(self, threshold_percentile: float = 90):
        self.threshold_percentile = threshold_percentile
        self.extractor = NextEventTransitionExtractor()
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit_extractor(self, df_train) -> "MarkovSequenceDetector":
        """Learn the normal transition probabilities from the training events.

        Must be called before ``fit``; it is the raw-event side of the detector
        (the standard ``fit`` receives the already-extracted feature matrix).
        """
        self.extractor.fit(df_train)
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # Mean log-prob is negative (logs of probabilities); negating makes a
        # high score = rare/unlikely transitions = anomalous.
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
        if self.score_min is None or self.score_max is None:
            self._check_fitted("score_min/score_max")

        X = self._to_float(X)
        raw = self._raw_scores(X)
        scores = self._scores_to_unit(raw, self.score_min, self.score_max)

        anomalies = self._above_threshold(scores, self.threshold)

        return anomalies, scores


if __name__ == "__main__":
    # Fase 3 DoD: a day whose transitions are rare under the Fase-1 movement graph
    # must score high; a typical generated day must score low.
    import pandas as pd

    from ingestion.markov_generator import (
        build_movement_graph,
        generate_daily_events,
    )

    rng = np.random.default_rng(7)
    graph = build_movement_graph({"door_activity": 1.0})
    start = pd.Timestamp("2024-01-01")

    def stream_of_day(day_offset: int, reverse: bool = False) -> pd.DataFrame:
        rows = generate_daily_events(
            rng, start + pd.Timedelta(days=day_offset), 180, 0.08, graph,
            {"name": "typical", "event_factor": 1.0, "night_offset": 0.0},
        )
        if reverse:
            # Reverse the intra-day order: keeps the sensor multiset, inverts the
            # transitions (rare when the graph is asymmetric).
            rows = rows[::-1]
        df = pd.DataFrame(rows, columns=["date", "time", "sensor_id", "reading"])
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.sort_values("timestamp")
        return df[["timestamp", "sensor_id"]]

    train = pd.concat([stream_of_day(d) for d in range(30)])
    typical = stream_of_day(31)
    rare = stream_of_day(32, reverse=True)

    detector = MarkovSequenceDetector()
    detector.fit_extractor(train)
    X_train = detector.extractor.extract(train)[0]
    detector.fit(X_train)

    _, score_typical = detector.predict(detector.extractor.extract(typical)[0])
    _, score_rare = detector.predict(detector.extractor.extract(rare)[0])

    print(f" Typical-day score: {float(score_typical[0]):.3f}")
    print(f" Reversed-day score: {float(score_rare[0]):.3f}")
    assert float(score_rare[0]) > float(score_typical[0]), (
        "rare transitions must score higher than typical ones"
    )
    detector._assert_unit_range(score_rare)
    print(" Validation OK")