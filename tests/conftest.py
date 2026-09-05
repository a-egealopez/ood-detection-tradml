import numpy as np
import pandas as pd
import pytest

from ingestion.casas_stream_generator import (
    build_weighted_graph,
    simulate_day,
    simulate_house,
)

RNG_SEED = 7


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def movement_graph() -> dict:
    return build_weighted_graph({"door_activity": 1.0})


def _events_markov(n_days: int, rng, graph) -> pd.DataFrame:
    """A normal multi-day stream in the injector/loader CSV schema."""
    days = []
    for d in range(n_days):
        rows = simulate_day(
            rng,
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
            180,
            0.08,
            graph,
            {"name": "typical", "event_factor": 1.0, "night_offset": 0.0},
        )
        days.append(rows)
    flat = [row for day in days for row in day]
    df = pd.DataFrame(flat, columns=["date", "time", "sensor_id", "reading"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_type"] = "ON"
    df["value"] = 1.0
    return df[["timestamp", "sensor_id", "event_type", "value"]]


@pytest.fixture
def casas_stream(rng, movement_graph) -> pd.DataFrame:
    """A normal 40-day Markov-generated event stream (asymmetric transitions)."""
    return _events_markov(40, rng, movement_graph)


@pytest.fixture
def house_stream() -> pd.DataFrame:
    """A full 60-day house stream in raw CSV schema (date/time/sensor/reading)."""
    return simulate_house(
        house_id="aruba", events_mean=180, night_ratio=0.08, n_days=60, seed=1
    )


@pytest.fixture
def normal_samples() -> np.ndarray:
    """5-D standard-normal training samples shared by vectorial detector tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 5))
