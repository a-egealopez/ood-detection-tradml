import numpy as np
import pandas as pd

from ingestion.casas_stream_generator import (
    build_weighted_graph,
    get_band,
    simulate_house,
)


# DoD gate functions (moved from source to keep source clean)
def graph_entropy_ratio(graph: dict) -> float:
    """Mean transition entropy over (band, source) rows, normalized by log2(n_sensors).

    DoD gate requires < 0.70 (peaked graph, not uniform).
    """
    from ingestion.casas_stream_generator import SENSORS, normalize_weights_to_probabilities
    n_sensors = len(SENSORS)
    ratios = []
    for band in range(4):
        for source in SENSORS:
            probs = np.array(list(normalize_weights_to_probabilities(graph, band, source).values()))
            probs = probs[probs > 0]
            entropy_bits = float(-(probs * np.log2(probs)).sum())
            ratios.append(entropy_bits / np.log2(n_sensors))
    return float(np.mean(ratios))


def first_order_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient of a 1-D series (Pearson)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return 0.0
    x, y = values[:-1], values[1:]
    x_std, y_std = x.std(), y.std()
    if x_std == 0 or y_std == 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x_std * y_std))


def test_graph_entropy_ratio_below_gate():
    """Generator gate: the movement graph must be peaked (structured), not uniform."""
    graph = build_weighted_graph({"door_activity": 1.0})
    assert graph_entropy_ratio(graph) < 0.70


def test_house_stream_has_lag1_autocorrelation():
    """Generator gate: daily features must carry lag-1 autocorrelation (>= 0.30)."""
    stream = simulate_house(
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


def test_get_band_mapping():
    assert get_band(8) == 0  # morning
    assert get_band(12) == 1  # midday
    assert get_band(17) == 2  # afternoon
    assert get_band(23) == 3  # night
    assert get_band(3) == 3  # night
