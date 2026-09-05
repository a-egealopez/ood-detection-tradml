import numpy as np
import pandas as pd

from features.daily_aggregates import daily_aggregates, entropy, extract_by_date


def test_entropy_uniform_histogram_uses_full_range():
    counts = np.array([1, 1, 1, 1])
    assert np.isclose(entropy(counts), np.log(4), atol=1e-6)


def test_entropy_empty_returns_zero():
    assert entropy(np.array([0, 0, 0])) == 0.0


def test_entropy_zero_total_returns_zero():
    assert entropy(np.array([0, 0])) == 0.0


def test_entropy_concentrated_is_lower():
    assert entropy(np.array([100, 0, 0, 0])) < entropy(np.array([25, 25, 25, 25]))


def test_daily_aggregates_empty_group():
    group = pd.DataFrame(
        {"timestamp": pd.Series([], dtype="datetime64[ns]"), "sensor_id": [], "hour": []}
    )
    agg = daily_aggregates(group, include_peak_hour=True, include_frequency_std=True)
    assert agg["n_events"] == 0
    assert agg["n_sensors"] == 0


def test_daily_aggregates_counts_and_entropy():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-01 08:30", "2024-01-01 09:00", "2024-01-01 23:00"]
            ),
            "sensor_id": ["Bedroom", "Kitchen", "Kitchen", "Bedroom"],
            "hour": [8, 8, 9, 23],
        }
    )
    agg = daily_aggregates(df, include_peak_hour=True, include_frequency_std=True)
    assert agg["n_events"] == 4
    assert agg["n_sensors"] == 2
    assert agg["peak_hour"] == 8
    assert np.isclose(agg["entropy_sensor"], np.log(2), atol=1e-6)


def test_extract_by_date_returns_rows_and_dates():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-01 09:00", "2024-01-02 08:00"]
            ),
            "sensor_id": ["A", "A", "A"],
        }
    )
    X, dates = extract_by_date(df, lambda g: [len(g)])
    assert X.shape == (2, 1)
    assert len(dates) == 2
