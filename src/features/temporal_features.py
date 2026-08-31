from typing import ClassVar

import numpy as np
import pandas as pd

from features.common import daily_aggregates, extract_by_date


class TemporalFeatureExtractor:
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

    def _features_for_group(self, group: pd.DataFrame) -> list:
        aggregates = daily_aggregates(
            group,
            include_peak_hour=True,
            include_frequency_std=True,
        )
        return [aggregates[key] for key in self._ORDER]
