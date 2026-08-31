"""
Per-window (daily) feature extractors for event-based time series.

Three extractors, each targeting different anomaly types (Chandola et al., 2009
survey taxonomy): point, contextual, collective/sequence.

1. WindowAggregationExtractor   -> contextual + collective (day level)
2. IntervalStatisticsExtractor  -> point (raw intervals) or collective (per window)
3. NGramTransitionExtractor     -> collective/sequence (pattern-based)

Uniform interface: extract(df) -> (X, dates) and diagnostics(group) -> dict.
"""

import itertools
from typing import ClassVar

import numpy as np
import pandas as pd

from config import DEFAULT_RANDOM_STATE
from features.common import EPSILON, daily_aggregates, entropy, extract_by_date
from ingestion.markov_generator import NOISE


class WindowAggregationExtractor:
    """Statistical per-window aggregation.

    Anomaly type: contextual (hour context encoded in features) + collective at
    day level. Not suited to point anomalies (one rare event in a normal day).
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "n_events",
        "n_sensors",
        "activity_hours",
        "avg_gap_minutes",
        "night_activity_ratio",
        "entropy_hourly",
        "entropy_sensor",
    ]

    # Maps this extractor's feature names to the keys of the shared helper.
    _ORDER: ClassVar[list[str]] = [
        "n_events",
        "n_sensors",
        "activity_hours",
        "avg_gap_minutes",
        "night_activity",
        "entropy_hourly",
        "entropy_sensor",
    ]

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return extract_by_date(df, self._features_for_window)

    def _features_for_window(self, group: pd.DataFrame) -> list[float]:
        aggregates = daily_aggregates(group, base=2)
        return [aggregates[key] for key in self._ORDER]

    def diagnostics(self, group: pd.DataFrame) -> dict:
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        group["hour"] = group["timestamp"].dt.hour
        hourly_counts = group.groupby("hour").size().reindex(range(24), fill_value=0)
        return {
            "hourly_counts": hourly_counts,
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


class IntervalStatisticsExtractor:
    """Interval statistics between consecutive events (point-process / renewal).

    Anomaly type: point if raw intervals are used (``diagnostics()``), collective
    if the per-window aggregate from ``extract()`` is used.
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "n_events",
        "mean_iei_sec",
        "std_iei_sec",
        "cv_iei",
        "fano_factor",
    ]

    def __init__(self, fano_bin_minutes: float = 30.0):
        self.fano_bin_minutes = fano_bin_minutes

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return extract_by_date(df, self._features_for_window)

    def _intervals_seconds(self, group: pd.DataFrame) -> np.ndarray:
        ts_sorted = pd.to_datetime(group["timestamp"]).sort_values()
        return ts_sorted.diff().dropna().dt.total_seconds().values

    def _fano_factor(self, group: pd.DataFrame) -> float:
        """Variance / mean of the event count in fixed-size bins: measures 'burstiness'."""
        ts = pd.to_datetime(group["timestamp"]).sort_values()
        if len(ts) < 2:
            return 0.0
        bin_edges = pd.date_range(
            ts.min().floor("min"),
            ts.max().ceil("min") + pd.Timedelta(minutes=self.fano_bin_minutes),
            freq=f"{self.fano_bin_minutes}min",
        )
        if len(bin_edges) < 2:
            return 0.0
        counts, _ = np.histogram(ts.astype("int64"), bins=bin_edges.astype("int64"))
        mean_count = counts.mean()
        if mean_count == 0:
            return 0.0
        return float(counts.var() / mean_count)

    def _features_for_window(self, group: pd.DataFrame) -> list[float]:
        n_events = len(group)
        intervals = self._intervals_seconds(group)
        if len(intervals) < 2:
            return [float(n_events), 0.0, 0.0, 0.0, 0.0]

        mean_iei = float(intervals.mean())
        std_iei = float(intervals.std())
        cv_iei = float(std_iei / (mean_iei + EPSILON))
        fano = self._fano_factor(group)
        return [float(n_events), mean_iei, std_iei, cv_iei, fano]

    def diagnostics(self, group: pd.DataFrame) -> dict:
        return {
            "intervals_seconds": self._intervals_seconds(group),
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


class NGramTransitionExtractor:
    """First-order Markov chain over the sequence of triggered sensors.

    Anomaly type: collective/sequence (pattern-based). Only looks at event order,
    so it cannot detect point or pure contextual anomalies.
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "n_transitions",
        "transition_entropy",
        "top_transition_prob",
        "unique_bigrams_ratio",
    ]

    def __init__(self, token_col: str = "sensor_id"):  # noqa: S107
        self.token_col = token_col
        self.vocabulary_: list[str] = []

    def fit_vocabulary(self, df: pd.DataFrame) -> "NGramTransitionExtractor":
        self.vocabulary_ = sorted(df[self.token_col].astype(str).unique().tolist())
        return self

    def _sequence(self, group: pd.DataFrame) -> list[str]:
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        return group.sort_values("timestamp")[self.token_col].astype(str).tolist()

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.vocabulary_:
            self.fit_vocabulary(df)
        return extract_by_date(df, self._features_for_window)

    def _transition_matrix(self, group: pd.DataFrame) -> pd.DataFrame:
        sequence = self._sequence(group)
        vocabulary = self.vocabulary_ or sorted(set(sequence))
        matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary, dtype=float)
        for source, target in itertools.pairwise(sequence):
            if source in matrix.index and target in matrix.columns:
                matrix.loc[source, target] += 1
        return matrix

    def _features_for_window(self, group: pd.DataFrame) -> list[float]:
        sequence = self._sequence(group)
        n_transitions = max(len(sequence) - 1, 0)
        if n_transitions == 0:
            return [0.0, 0.0, 0.0, 0.0]

        matrix = self._transition_matrix(group)
        transition_counts = matrix.values.flatten()
        transition_entropy = entropy(transition_counts, base=2)

        total_transitions = transition_counts.sum()
        top_transition_prob = (
            float(transition_counts.max() / total_transitions)
            if total_transitions > 0
            else 0.0
        )

        possible_bigrams = len(self.vocabulary_) ** 2 if self.vocabulary_ else 1
        unique_bigrams = int((transition_counts > 0).sum())
        unique_bigrams_ratio = float(unique_bigrams / possible_bigrams)

        return [
            float(n_transitions),
            transition_entropy,
            top_transition_prob,
            unique_bigrams_ratio,
        ]

    def diagnostics(self, group: pd.DataFrame) -> dict:
        return {
            "transition_matrix": self._transition_matrix(group),
            "sequence": self._sequence(group),
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


class NextEventTransitionExtractor:
    """First-order Markov *prediction* of the next sensor.

    Anomaly type: point (single event) + collective at day level. Unlike
    ``NGramTransitionExtractor`` (which summarizes a day into an entropy), this
    learns a transition-probability matrix from normal behavior and scores each
    real transition by its log-likelihood (DeepLog-style), so one unlikely step in
    an otherwise normal day is not diluted by aggregation.

    Fit uses ML estimation with Laplace smoothing (no zero probabilities); the day
    signal is the negative log-likelihood of its real transitions.
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "mean_logprob",
        "min_logprob",
        "rare_transition_rate",
    ]

    # Relative (not absolute) so it works for any vocabulary size.
    RARE_Z: ClassVar[float] = 2.0
    # One pseudo-count per (source, target) pair.
    LAPLACE_ALPHA: ClassVar[float] = 1.0

    def __init__(self, token_col: str = "sensor_id"):  # noqa: S107
        self.token_col = token_col
        self.prob_matrix_: pd.DataFrame | None = None
        self.rare_threshold_: float | None = None

    def fit(self, df: pd.DataFrame) -> "NextEventTransitionExtractor":
        """Learn the normal-transition probability matrix from ``df``."""
        sequence = self._sequence(df)
        vocabulary = sorted(set(sequence))
        transition_counts = pd.DataFrame(
            self.LAPLACE_ALPHA, index=vocabulary, columns=vocabulary, dtype=float
        )
        for source, target in itertools.pairwise(sequence):
            if source in transition_counts.index and target in transition_counts.columns:
                transition_counts.loc[source, target] += 1.0
        self.prob_matrix_ = transition_counts.div(
            transition_counts.sum(axis=1), axis=0
        )

        train_logprobs = self._logprobs_of(sequence)
        self.rare_threshold_ = (
            float(np.mean(train_logprobs))
            - self.RARE_Z * float(np.std(train_logprobs))
            if len(train_logprobs) > 1
            else -float("inf")
        )
        return self

    def _sequence(self, df: pd.DataFrame) -> list[str]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")[self.token_col].astype(str).tolist()

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.prob_matrix_ is None:
            self.fit(df)
        return extract_by_date(df, self._features_for_window)

    def _logprobs_of(self, seq: list[str]) -> list[float]:
        prob_matrix = self.prob_matrix_
        logprobs = []
        for source, target in itertools.pairwise(seq):
            if source in prob_matrix.index and target in prob_matrix.columns:
                probability = float(prob_matrix.loc[source, target])
                logprobs.append(float(np.log(max(probability, EPSILON))))
            else:
                # Source or target sensor never seen in training: maximally rare.
                logprobs.append(float(np.log(EPSILON)))
        return logprobs

    def _transition_logprobs(self, group: pd.DataFrame) -> list[float]:
        return self._logprobs_of(self._sequence(group))

    def _features_for_window(self, group: pd.DataFrame) -> list[float]:
        logprobs = self._transition_logprobs(group)
        n_transitions = len(logprobs)
        if n_transitions == 0:
            return [0.0, 0.0, 0.0]
        rare_transition_rate = (
            sum(1.0 for lp in logprobs if lp < self.rare_threshold_) / n_transitions
        )
        return [float(np.mean(logprobs)), float(np.min(logprobs)), rare_transition_rate]

    def diagnostics(self, group: pd.DataFrame) -> dict:
        logprobs = self._transition_logprobs(group)
        sequence = self._sequence(group)
        prob_matrix = self.prob_matrix_
        transitions = []
        for (source, target), logprob in zip(
            itertools.pairwise(sequence), logprobs, strict=True
        ):
            transitions.append(
                {
                    "from": source,
                    "to": target,
                    "prob": float(np.exp(logprob)) if logprob > np.log(EPSILON) else 0.0,
                    "logprob": logprob,
                    "rare": logprob < self.rare_threshold_,
                }
            )
        return {
            "sequence": sequence,
            "transition_matrix": prob_matrix.copy(),
            "transitions": transitions,
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
            "rare_threshold": self.rare_threshold_,
        }


def sensor_chain_probabilities(n_sensors: int) -> np.ndarray:
    """First-order transition matrix: asymmetric directed cycle (``i -> i+1`` dominant).

    Asymmetry makes a reversed day (collective injector) produce rare transitions;
    a symmetric stream would be indistinguishable after reversal. Falls back to
    uniform for ``n_sensors < 3`` (a cycle needs three states).
    """
    if n_sensors < 3:
        return np.full((n_sensors, n_sensors), 1.0 / n_sensors)
    weights = np.ones((n_sensors, n_sensors))
    for i in range(n_sensors):
        weights[i, i] = 5.0
        weights[i, (i + 1) % n_sensors] = 25.0
        weights[i, (i - 1) % n_sensors] = 2.0
    probs = weights / weights.sum(axis=1, keepdims=True)
    return (1.0 - NOISE) * probs + NOISE / n_sensors


def generate_synthetic_events(
    n_days: int = 5,
    pattern: str = "regular",
    n_sensors: int = 3,
    events_per_day: int = 80,
    seed: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Synthetic event stream in CASAS-Aruba schema (no DB needed).

    Sensors are drawn from an asymmetric first-order Markov chain (see
    ``sensor_chain_probabilities``) so the sequence extractors have structure to
    learn and a reversal produces rare transitions.

    pattern: "regular" (evenly spaced), "bursty" (temporal clusters), or
    "day_night" (mostly daytime).
    """
    rng = np.random.default_rng(seed)
    sensors = [f"Sensor_{i + 1}" for i in range(n_sensors)]
    transition_probs = sensor_chain_probabilities(n_sensors)
    base_day = pd.Timestamp("2024-01-01")
    records = []

    for day in range(n_days):
        day_start = base_day + pd.Timedelta(days=day)

        if pattern == "regular":
            offsets = np.linspace(0, 24 * 60, events_per_day, endpoint=False)
            offsets = offsets + rng.normal(0, 2, size=events_per_day)
        elif pattern == "bursty":
            n_clusters = max(3, events_per_day // 15)
            centers = rng.uniform(0, 24 * 60, size=n_clusters)
            per_cluster = max(events_per_day // n_clusters, 1)
            offsets = np.concatenate(
                [rng.normal(center, 5, size=per_cluster) for center in centers]
            )
        elif pattern == "day_night":
            day_events = int(events_per_day * 0.85)
            night_events = events_per_day - day_events
            offsets = np.concatenate(
                [
                    rng.uniform(8 * 60, 22 * 60, size=day_events),
                    rng.uniform(0, 8 * 60, size=night_events // 2),
                    rng.uniform(
                        22 * 60, 24 * 60, size=night_events - night_events // 2
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        offsets = np.clip(offsets, 0, 24 * 60 - 0.01)
        timestamps = [day_start + pd.Timedelta(minutes=float(minutes)) for minutes in offsets]
        seq = np.empty(len(timestamps), dtype=int)
        seq[0] = int(rng.integers(0, n_sensors))
        for i in range(1, len(timestamps)):
            seq[i] = int(rng.choice(n_sensors, p=transition_probs[seq[i - 1]]))
        chosen_sensors = [sensors[int(i)] for i in seq]
        event_types = rng.choice(["ON", "OFF"], size=len(timestamps))

        for timestamp, sensor, event_type in zip(
            timestamps, chosen_sensors, event_types, strict=True
        ):
            records.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor,
                    "event_type": event_type,
                    "value": 1.0 if event_type == "ON" else 0.0,
                }
            )

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
