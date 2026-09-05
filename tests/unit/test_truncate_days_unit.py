import pandas as pd
import pytest

from features.daily_aggregates import truncate_stream_to_days
from ingestion.casas_stream_generator import simulate_house


@pytest.fixture
def stream():
    raw = simulate_house(
        house_id="h", events_mean=400, night_ratio=0.1, n_days=20, seed=1
    )
    raw["timestamp"] = pd.to_datetime(raw["date"] + " " + raw["time"])
    return raw.drop(columns=["date", "time"])


def test_returns_all_days_when_max_days_none(stream):
    out = truncate_stream_to_days(stream, max_days=None)
    assert len(out) == len(stream)
    assert out.equals(stream)


def test_noop_when_max_days_exceeds_span(stream):
    out = truncate_stream_to_days(stream, max_days=10_000)
    assert len(out) == len(stream)


def test_keeps_only_first_n_days(stream):
    out = truncate_stream_to_days(stream, max_days=5)
    n_days = out["timestamp"].dt.date.nunique()
    assert n_days == 5
    dates = sorted(out["timestamp"].dt.date.unique())
    all_dates = sorted(pd.to_datetime(stream["timestamp"]).dt.date.unique())
    assert dates == all_dates[:5]


def test_no_events_from_later_days_leak(stream):
    out = truncate_stream_to_days(stream, max_days=5)
    all_dates = sorted(pd.to_datetime(stream["timestamp"]).dt.date.unique())
    later = pd.to_datetime(all_dates[-1]).date()
    assert later not in set(out["timestamp"].dt.date)


def test_preserves_event_columns(stream):
    out = truncate_stream_to_days(stream, max_days=5)
    assert set(out.columns) == set(stream.columns)
    assert out["timestamp"].notna().all()
