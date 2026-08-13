import numpy as np
import pandas as pd
from typing import ClassVar

EPSILON = 1e-10


def _calculate_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + EPSILON)))


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

    def __init__(self, window_size: int = 20, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap

    def extract(self, df: pd.DataFrame, group_by: str = "date"):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour

        features_list = []
        dates_list = []

        for date, group in df.groupby("date"):
            features_list.append(self._features_for_group(group))
            dates_list.append(date)

        X = np.array(features_list)
        dates = np.array(dates_list)
        return X, dates

    def _features_for_group(self, group: pd.DataFrame) -> list:
        n_events = len(group)

        if n_events == 0:
            return [0.0] * len(self.FEATURE_NAMES)

        n_sensors = group["sensor_id"].nunique()
        hours_active = group["hour"].nunique()

        ts_sorted = group["timestamp"].sort_values()
        if n_events > 1:
            gaps = ts_sorted.diff().dropna().dt.total_seconds() / 60.0
            avg_gap = float(gaps.mean())
        else:
            avg_gap = 0.0

        hour_mode = group["hour"].mode()
        peak_hour = int(hour_mode.iloc[0]) if len(hour_mode) > 0 else 12

        night_mask = (group["hour"] < 8) | (group["hour"] >= 22)
        night_activity = float(night_mask.sum()) / n_events

        events_per_sensor = group.groupby("sensor_id").size()
        event_frequency_std = float(events_per_sensor.std()) if n_sensors > 1 else 0.0

        hourly_counts = (
            group.groupby("hour").size().reindex(range(24), fill_value=0).values
        )
        entropy_hourly = _calculate_entropy(hourly_counts)
        entropy_sensor = _calculate_entropy(events_per_sensor.values)

        return [
            n_events,
            n_sensors,
            hours_active,
            avg_gap,
            peak_hour,
            night_activity,
            event_frequency_std,
            entropy_hourly,
            entropy_sensor,
        ]

    def rolling_features(self, df: pd.DataFrame, window_size: int | None = None):
        window_size = window_size or self.window_size

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) < window_size:
            raise ValueError(
                f"Se necesitan al menos {window_size} días distintos, "
                f"pero solo hay {len(unique_dates)}"
            )

        features_list = []
        window_end_dates = []

        for i in range(len(unique_dates) - window_size + 1):
            window_dates = unique_dates[i : i + window_size]
            window_df = df[df["date"].isin(window_dates)].copy()
            window_df["hour"] = window_df["timestamp"].dt.hour

            features_list.append(self._features_for_group(window_df))
            window_end_dates.append(window_dates[-1])

        X = np.array(features_list)
        return X, np.array(window_end_dates)


if __name__ == "__main__":
    np.random.seed(0)
    n = 200
    base_time = pd.Timestamp("2024-01-01")
    timestamps = [
        base_time + pd.Timedelta(minutes=int(m))
        for m in np.random.randint(0, 60 * 24 * 3, n)
    ]
    sensors = np.random.choice(["Bedroom", "Kitchen", "OutsideDoor"], size=n)
    event_types = np.random.choice(["ON", "OFF"], size=n)
    values = (event_types == "ON").astype(float)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": sensors,
            "event_type": event_types,
            "value": values,
        }
    )

    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df)

    print(f" Features shape: {X.shape}")
    print(f" Dates: {dates}")
    print(f" Feature names: {TemporalFeatureExtractor.FEATURE_NAMES}")
    print(pd.DataFrame(X, columns=TemporalFeatureExtractor.FEATURE_NAMES, index=dates))
