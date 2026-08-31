import numpy as np
import pandas as pd

from ingestion.markov_generator import (
    build_movement_graph,
    first_order_autocorrelation,
    generate_house_stream,
    graph_entropy_ratio,
    hour_to_band,
)


def test_graph_entropy_ratio_below_gate():
    """Generator gate: the movement graph must be peaked (structured), not uniform."""
    graph = build_movement_graph({"door_activity": 1.0})
    assert graph_entropy_ratio(graph) < 0.70


def test_house_stream_has_lag1_autocorrelation():
    """Generator gate: daily features must carry lag-1 autocorrelation (>= 0.30)."""
    stream = generate_house_stream(
        house_id="aruba", events_mean=180, night_ratio=0.08, n_days=60, seed=1
    )
    daily_counts = (
        pd.to_datetime(stream["date"]).value_counts().sort_index().astype(float).values
    )
    assert abs(first_order_autocorrelation(daily_counts)) >= 0.30


def test_first_order_autocorrelation_small_series_returns_zero():
    assert first_order_autocorrelation([1.0, 2.0]) == 0.0


def test_first_order_autocorrelation_constant_returns_zero():
    assert first_order_autocorrelation(np.ones(10)) == 0.0


def test_hour_to_band_mapping():
    assert hour_to_band(8) == 0  # morning
    assert hour_to_band(12) == 1  # midday
    assert hour_to_band(17) == 2  # afternoon
    assert hour_to_band(23) == 3  # night
    assert hour_to_band(3) == 3  # night
