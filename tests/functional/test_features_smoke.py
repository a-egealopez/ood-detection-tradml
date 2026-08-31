import numpy as np
import pandas as pd

from features.temporal_features import TemporalFeatureExtractor


def _synthetic_events(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01")
    timestamps = [
        base + pd.Timedelta(minutes=int(m))
        for m in rng.integers(0, 60 * 24 * 3, n)
    ]
    sensors = rng.choice(["Bedroom", "Kitchen", "OutsideDoor"], size=n)
    event_types = rng.choice(["ON", "OFF"], size=n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": sensors,
            "event_type": event_types,
            "value": (event_types == "ON").astype(float),
        }
    )


def test_extract_shape_and_feature_names():
    df = _synthetic_events()
    X, dates = TemporalFeatureExtractor().extract(df)
    assert X.shape[1] == len(TemporalFeatureExtractor.FEATURE_NAMES)
    assert len(X) == len(dates)


def test_count_columns_only_counts():
    X = np.zeros((5, 9))
    out = TemporalFeatureExtractor.count_columns(X)
    cols = [TemporalFeatureExtractor.FEATURE_NAMES.index(n)
            for n in TemporalFeatureExtractor.COUNT_FEATURE_NAMES]
    assert out.shape[1] == len(cols)


def test_hourly_counts_24_dim():
    df = _synthetic_events()
    X = TemporalFeatureExtractor.hourly_counts(df)
    assert X.shape[1] == 24
    assert (X >= 0).all()
