import numpy as np
import pandas as pd
from typing import ClassVar

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

    # Maps this extractor's feature names (the order above) to the keys
    # returned by the shared daily-aggregation helper.
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

    def extract(self, df: pd.DataFrame):
        return extract_by_date(df, self._features_for_group)

    def _features_for_group(self, group: pd.DataFrame) -> list:
        agg = daily_aggregates(
            group,
            include_peak_hour=True,
            include_frequency_std=True,
        )
        return [agg[key] for key in self._ORDER]


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
