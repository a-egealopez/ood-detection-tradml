"""
Per-window (daily) feature extractors for event-based time series.

Four extractors, each targeting different anomaly types (Chandola et al., 2009
survey taxonomy): point, contextual, collective/sequence.

1. TemporalFeatureExtractor      -> contextual + collective (day level)
2. IntervalStatisticsExtractor   -> point (raw intervals) or collective (per window)
3. NGramTransitionExtractor      -> collective/sequence (pattern-based)
4. NextEventTransitionExtractor  -> point (single event) + collective (day level)

Uniform interface: extract(df) -> (X, dates) and diagnostics(group) -> dict.
"""

import itertools
from typing import ClassVar

import numpy as np
import pandas as pd

from config import DEFAULT_RANDOM_STATE
from features.common import (
    EPSILON,
    daily_aggregates,
    entropy,
    event_sequence,
    extract_by_date,
)
from ingestion.markov_generator import NOISE


class EventDrivenExtractor:
    """Base class for the didactic daily extractors.

    Centralizes the shared interface every extractor must expose:

    - ``extract(df) -> (X, dates)``: one feature row per day, built by
      ``_features_for_group``.
    - ``FEATURE_NAMES``: the (class) names of the features in each row.
    - ``diagnostics(group) -> dict``: what a single day reveals, for plotting.

    Subclasses only implement ``_features_for_group`` and (optionally)
    override ``diagnostics`` with their method-specific visuals.
    """

    FEATURE_NAMES: ClassVar[list[str]]

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return extract_by_date(df, self._features_for_group)

    def _features_for_group(self, group: pd.DataFrame) -> list[float]:
        raise NotImplementedError

    def diagnostics(self, group: pd.DataFrame) -> dict:
        return {
            "features": self._features_for_group(group),
            "feature_names": self.FEATURE_NAMES,
        }


class TemporalFeatureExtractor:
    """Daily-aggregation extractor used by the production CASAS pipeline.

    Reduces each day to a 9-feature vector of counts, spreads and entropies
    (contextual + collective at day level). Anomaly type: contextual (hour
    context encoded in the features) + collective at day level. Not suited to
    point anomalies (one rare event in a normal day).
    """

    FEATURE_NAMES: ClassVar[list[str]] = [
        "n_events",
        "n_sensors",
        "activity_hours",
        "avg_event_gap_minutes",
        "peak_hour",
        "night_activity",
        "event_frequency_std",
        "entropy_hourly",
        "entropy_sensor",
    ]

    # Count-feature subset shared with COUNT_FEATURE_NAMES order. Only these are
    # valid for the Hawkes detector: its Poisson intensity model needs integer counts.
    COUNT_FEATURE_NAMES: ClassVar[list[str]] = [
        "n_events",
        "n_sensors",
        "activity_hours",
    ]

    # Maps FEATURE_NAMES onto the keys returned by the shared aggregator.
    _ORDER: ClassVar[list[str]] = [
        "n_events",
        "n_sensors",
        "activity_hours",
        "avg_gap_minutes",
        "peak_hour",
        "night_activity",
        "event_frequency_std",
        "entropy_hourly",
        "entropy_sensor",
    ]

    @classmethod
    def count_columns(cls, X) -> np.ndarray:
        """Return just the count-feature columns (indices from COUNT_FEATURE_NAMES)."""
        indices = [cls.FEATURE_NAMES.index(name) for name in cls.COUNT_FEATURE_NAMES]
        return np.asarray(X, dtype=float)[:, indices]

    @classmethod
    def hourly_counts(cls, df) -> np.ndarray:
        """Per-day 24-element hourly event counts (row order matches ``extract``).

        Hourly resolution makes the Hawkes detector sensitive to *when* events
        happen (contextual anomalies) while blind to intra-day order (collective).
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        rows = [
            group.groupby("hour").size().reindex(range(24), fill_value=0).values
            for _, group in df.groupby("date")
        ]
        return np.asarray(rows, dtype=float)

    def extract(self, df: pd.DataFrame):
        return extract_by_date(df, self._features_for_group)

    def diagnostics(self, group: pd.DataFrame) -> dict:
        """Expose one day's bag of features, for the didactic inspector."""
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        group["hour"] = group["timestamp"].dt.hour
        return {
            "features": self._features_for_group(group),
            "feature_names": self.FEATURE_NAMES,
        }

    def _features_for_group(self, group: pd.DataFrame) -> list:
        aggregates = daily_aggregates(
            group,
            include_peak_hour=True,
            include_frequency_std=True,
        )
        return [aggregates[key] for key in self._ORDER]


class IntervalStatisticsExtractor(EventDrivenExtractor):
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

    def _features_for_group(self, group: pd.DataFrame) -> list[float]:
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
            "features": self._features_for_group(group),
            "feature_names": self.FEATURE_NAMES,
        }


class NGramTransitionExtractor(EventDrivenExtractor):
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

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.vocabulary_:
            self.fit_vocabulary(df)
        return super().extract(df)

    def _transition_matrix(self, group: pd.DataFrame) -> pd.DataFrame:
        sequence = event_sequence(group, self.token_col)
        vocabulary = self.vocabulary_ or sorted(set(sequence))
        matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary, dtype=float)
        for source, target in itertools.pairwise(sequence):
            if source in matrix.index and target in matrix.columns:
                matrix.loc[source, target] += 1
        return matrix

    def _features_for_group(self, group: pd.DataFrame) -> list[float]:
        sequence = event_sequence(group, self.token_col)
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
            "sequence": event_sequence(group, self.token_col),
            "features": self._features_for_group(group),
            "feature_names": self.FEATURE_NAMES,
        }


class NextEventTransitionExtractor(EventDrivenExtractor):
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

        # Speed path for large streams: integer codes + a stacked numpy matrix.
        self._vocab_index_: dict[str, int] = {}
        self._prob_values_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "NextEventTransitionExtractor":
        """Learn the normal-transition probability matrix from ``df``.

        Vectorized: the transition-count matrix is built with a numpy occurrence
        count over the integer-encoded sequence instead of a Python loop, so
        fitting scales to hundreds of thousands of raw events (real CASAS days)
        instead of quadratic ``DataFrame.loc`` writes.
        """
        sequence = event_sequence(df, self.token_col)
        vocabulary = sorted(set(sequence))
        n = len(vocabulary)
        self._vocab_index_ = {sensor: i for i, sensor in enumerate(vocabulary)}

        transition_counts = np.full(
            (n, n), self.LAPLACE_ALPHA, dtype=float
        )
        if n > 0 and len(sequence) > 1:
            codes = np.fromiter(
                (self._vocab_index_[s] for s in sequence),
                dtype=int,
                count=len(sequence),
            )
            flat = codes[:-1] * n + codes[1:]
            transition_counts += np.bincount(flat, minlength=n * n).reshape(n, n)

        self._prob_values_ = transition_counts / transition_counts.sum(
            axis=1, keepdims=True
        )
        self.prob_matrix_ = pd.DataFrame(
            self._prob_values_, index=vocabulary, columns=vocabulary
        )

        train_logprobs = self._logprobs_of(sequence)
        self.rare_threshold_ = (
            float(np.mean(train_logprobs))
            - self.RARE_Z * float(np.std(train_logprobs))
            if len(train_logprobs) > 1
            else -float("inf")
        )
        return self

    def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.prob_matrix_ is None:
            self.fit(df)
        return super().extract(df)

    def _logprobs_of(self, seq: list[str]) -> list[float]:
        prob_values = self._prob_values_
        if prob_values is None:
            raise RuntimeError("You must call fit() before extract()")
        n_transitions = max(len(seq) - 1, 0)
        if n_transitions == 0:
            return []

        codes = np.fromiter(
            (self._vocab_index_.get(s, -1) for s in seq),
            dtype=int,
            count=len(seq),
        )
        source = codes[:-1]
        target = codes[1:]
        known = (source >= 0) & (target >= 0)

        logprobs = np.full(n_transitions, float(np.log(EPSILON)), dtype=float)
        logprobs[known] = np.log(
            np.maximum(prob_values[source[known], target[known]], EPSILON)
        )
        return logprobs.tolist()

    def _transition_logprobs(self, group: pd.DataFrame) -> list[float]:
        return self._logprobs_of(event_sequence(group, self.token_col))

    def _features_for_group(self, group: pd.DataFrame) -> list[float]:
        logprobs = self._transition_logprobs(group)
        n_transitions = len(logprobs)
        if n_transitions == 0:
            return [0.0, 0.0, 0.0]
        rare_threshold = self.rare_threshold_
        if rare_threshold is None:
            raise RuntimeError("You must call fit() before extract()")
        rare_transition_rate = (
            sum(1.0 for lp in logprobs if lp < rare_threshold) / n_transitions
        )
        return [float(np.mean(logprobs)), float(np.min(logprobs)), rare_transition_rate]

    def diagnostics(self, group: pd.DataFrame) -> dict:
        logprobs = self._transition_logprobs(group)
        sequence = event_sequence(group, self.token_col)
        prob_matrix = self.prob_matrix_
        rare_threshold = self.rare_threshold_
        if prob_matrix is None or rare_threshold is None:
            raise RuntimeError("You must call fit() before extract()")
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
                    "rare": logprob < rare_threshold,
                }
            )
        return {
            "sequence": sequence,
            "transition_matrix": prob_matrix.copy(),
            "transitions": transitions,
            "features": self._features_for_group(group),
            "feature_names": self.FEATURE_NAMES,
            "rare_threshold": rare_threshold,
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
