"""Markov-chain event generator for synthetic CASAS-style data.

The fixtures must have structure for the sequential detectors to learn:

1. A directed movement graph per house with strongly asymmetric transition weights
   that depend on the hour band. Reversing a day's order (collective injector) only
   creates rare transitions when the common ones are not symmetric, so asymmetry is
   mandatory.
2. A sticky latent day regime (P(stay) ~ 0.75) that scales daily volume and night
   ratio, giving the daily features day-to-day autocorrelation for HMM/regime models.

Output rows use the exact loader CSV schema (``date, time, sensor_id, reading``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS: list[str] = ["Bedroom", "Kitchen", "LivingRoom", "OutsideDoor"]
MOTION_SENSORS = ("Bedroom", "Kitchen", "LivingRoom")

# Hour bands the movement graph conditions on.
BAND_MORNING, BAND_MIDDAY, BAND_AFTERNOON, BAND_NIGHT = range(4)
BAND_NAMES = ("morning", "midday", "afternoon", "night")

# Small uniform mix so the graph's structure dominates (else a reversal has nothing rare).
NOISE = 0.10

# Sticky latent day regimes: scale daily event count and night ratio.
REGIMES: list[dict] = [
    {"name": "quiet", "event_factor": 0.70, "night_offset": -0.02},
    {"name": "typical", "event_factor": 1.00, "night_offset": 0.00},
    {"name": "active", "event_factor": 1.35, "night_offset": 0.04},
]
REGIME_STAY_PROB = 0.75

# Diurnal activity curve for day-time hours (8-21): morning peak, midday lull,
# evening peak. Keeps hourly models sensitive to a circular shift (contextual).
DAY_HOUR_WEIGHTS: list[float] = [
    0.90, 1.30, 1.10, 1.00, 0.95, 0.85, 0.90, 1.00, 1.05, 1.20, 1.35, 1.25, 1.15, 1.00,
]  # hours 8..21


def hour_to_band(hour: int) -> int:
    """Map an hour-of-day to its movement-graph band."""
    if 6 <= hour < 11:
        return BAND_MORNING
    if 11 <= hour < 15:
        return BAND_MIDDAY
    if 15 <= hour < 20:
        return BAND_AFTERNOON
    return BAND_NIGHT


# ``weights[band][source][target]``, higher = more likely. One dominant target per
# source row. The rooms form a directed cycle (Bedroom -> Kitchen -> LivingRoom ->
# Bedroom) with backward edges near zero; this asymmetry makes reversed days
# (collective injector) produce rare transitions, which is what exposes them.
_BASE_WEIGHTS: dict[int, dict[str, dict[str, float]]] = {
    BAND_MORNING: {
        "Bedroom": {"Bedroom": 5.0, "Kitchen": 30.0, "LivingRoom": 1.0, "OutsideDoor": 1.0},
        "Kitchen": {"Kitchen": 5.0, "LivingRoom": 28.0, "Bedroom": 1.0, "OutsideDoor": 1.0},
        "LivingRoom": {"LivingRoom": 3.0, "Kitchen": 2.0, "Bedroom": 20.0, "OutsideDoor": 2.0},
        "OutsideDoor": {"OutsideDoor": 1.0, "Kitchen": 18.0, "LivingRoom": 6.0, "Bedroom": 1.0},
    },
    BAND_MIDDAY: {
        "Bedroom": {"Bedroom": 2.0, "Kitchen": 18.0, "LivingRoom": 4.0, "OutsideDoor": 1.0},
        "Kitchen": {"Kitchen": 6.0, "LivingRoom": 26.0, "Bedroom": 1.0, "OutsideDoor": 1.0},
        "LivingRoom": {"LivingRoom": 5.0, "Kitchen": 2.0, "Bedroom": 18.0, "OutsideDoor": 1.0},
        "OutsideDoor": {"OutsideDoor": 1.0, "Kitchen": 6.0, "LivingRoom": 16.0, "Bedroom": 1.0},
    },
    BAND_AFTERNOON: {
        "Bedroom": {"Bedroom": 2.0, "Kitchen": 24.0, "LivingRoom": 3.0, "OutsideDoor": 3.0},
        "Kitchen": {"Kitchen": 6.0, "LivingRoom": 24.0, "Bedroom": 1.0, "OutsideDoor": 2.0},
        "LivingRoom": {"LivingRoom": 6.0, "Kitchen": 2.0, "Bedroom": 20.0, "OutsideDoor": 1.0},
        "OutsideDoor": {"OutsideDoor": 1.0, "Kitchen": 18.0, "LivingRoom": 6.0, "Bedroom": 2.0},
    },
    BAND_NIGHT: {
        "Bedroom": {"Bedroom": 12.0, "Kitchen": 3.0, "LivingRoom": 1.0, "OutsideDoor": 1.0},
        "Kitchen": {"Kitchen": 3.0, "LivingRoom": 16.0, "Bedroom": 2.0, "OutsideDoor": 1.0},
        "LivingRoom": {"LivingRoom": 8.0, "Kitchen": 1.0, "Bedroom": 20.0, "OutsideDoor": 1.0},
        "OutsideDoor": {"OutsideDoor": 1.0, "Kitchen": 2.0, "LivingRoom": 16.0, "Bedroom": 4.0},
    },
}


def build_movement_graph(profile: dict) -> dict[tuple[int, str], dict[str, float]]:
    """Directed movement graph: ``(band, source) -> {target: weight}``.

    ``profile["door_activity"]`` (default 1.0) scales weights into/out of
    OutsideDoor, giving each house a distinct routine while keeping asymmetry.
    """
    door_factor = float(profile.get("door_activity", 1.0))
    graph: dict[tuple[int, str], dict[str, float]] = {}
    for band, band_weights in _BASE_WEIGHTS.items():
        for source, targets in band_weights.items():
            row: dict[str, float] = {}
            for target, weight in targets.items():
                scaled = weight * door_factor if "OutsideDoor" in (source, target) else weight
                row[target] = scaled
            graph[(band, source)] = row
    return graph


def transition_probabilities(graph, band: int, source: str) -> dict[str, float]:
    """Normalized ``{target: probability}`` for a (band, source) row."""
    weights = graph[(band, source)]
    total = sum(weights.values())
    return {target: w / total for target, w in weights.items()}


def graph_entropy_ratio(graph: dict) -> float:
    """Mean transition entropy over (band, source) rows, normalized by log2(n_sensors).

    DoD gate requires < 0.70 (peaked graph, not uniform).
    """
    n_sensors = len(SENSORS)
    ratios = []
    for band in range(4):
        for source in SENSORS:
            probs = np.array(list(transition_probabilities(graph, band, source).values()))
            probs = probs[probs > 0]
            entropy_bits = float(-(probs * np.log2(probs)).sum())
            ratios.append(entropy_bits / np.log2(n_sensors))
    return float(np.mean(ratios))


def draw_next_sensor(rng, prev_sensor: str, hour: int, graph: dict, noise: float = NOISE) -> str:
    """Sample the next sensor from the graph given the previous sensor and hour band."""
    band = hour_to_band(hour)
    probs = np.array([graph[(band, prev_sensor)][target] for target in SENSORS], dtype=float)
    probs = probs / probs.sum()
    mixed = (1.0 - noise) * probs + noise / len(SENSORS)
    return SENSORS[int(rng.choice(len(SENSORS), p=mixed))]


def draw_regime_sequence(rng, n_days: int) -> list[dict]:
    """Draw the latent day regime for each day (sticky Markov chain)."""
    states = []
    current = int(rng.integers(0, len(REGIMES)))
    for _ in range(n_days):
        states.append(REGIMES[current])
        if rng.random() >= REGIME_STAY_PROB:
            current = (current + 1 + int(rng.integers(0, len(REGIMES) - 1))) % len(REGIMES)
    return states


def _reading_for(sensor: str, rng) -> str:
    if sensor == "OutsideDoor":
        return rng.choice(["OPEN", "CLOSE"])
    return rng.choice(["ON", "OFF"])


def generate_daily_events(
    rng,
    date,
    events_mean: float,
    night_ratio: float,
    graph: dict,
    regime: dict,
    noise: float = NOISE,
) -> list[tuple[str, str, str]]:
    """One day of events as ``(time_str, sensor_id, reading)`` tuples.

    Draw timestamps (day/night hours per ``night_ratio``), then walk the graph to
    assign each event a sensor conditioned on its hour band.
    """
    n_events = max(1, int(rng.poisson(events_mean * regime["event_factor"])))
    night_ratio = min(0.5, max(0.01, night_ratio + regime["night_offset"]))

    is_night = rng.random(n_events) < night_ratio
    n_night = int(is_night.sum())
    hours = np.empty(n_events, dtype=int)
    if n_night:
        hours[is_night] = rng.choice(list(range(22, 24)) + list(range(8)), size=n_night)
    n_day = n_events - n_night
    if n_day:
        day_hours = np.arange(8, 22)
        day_probs = np.asarray(DAY_HOUR_WEIGHTS, dtype=float)
        day_probs = day_probs / day_probs.sum()
        hours[~is_night] = rng.choice(day_hours, size=n_day, p=day_probs)
    minutes = rng.integers(0, 60, size=n_events)
    seconds = rng.integers(0, 60, size=n_events)
    micros = rng.integers(0, 1_000_000, size=n_events)

    timestamps = [
        pd.Timestamp(date)
        + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s), microseconds=int(u))
        for h, m, s, u in zip(hours, minutes, seconds, micros, strict=True)
    ]
    timestamps.sort()

    # Markov walk over the movement graph, conditioned on each event's hour.
    first = SENSORS[int(rng.integers(0, len(MOTION_SENSORS)))]
    sensors = [first]
    for i in range(1, n_events):
        sensors.append(draw_next_sensor(rng, sensors[-1], timestamps[i].hour, graph, noise))

    rows = []
    for ts, sensor in zip(timestamps, sensors, strict=True):
        rows.append(
            (
                ts.strftime("%Y-%m-%d"),
                ts.strftime("%H:%M:%S.%f"),
                sensor,
                _reading_for(sensor, rng),
            )
        )
    return rows


def generate_house_stream(
    *,
    house_id: str,  # noqa: ARG001 - part of the public API signature
    events_mean: int,
    night_ratio: float,
    n_days: int,
    seed: int,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Full event stream for one house as a DataFrame in CSV schema.

    Draws per-day regimes, then walks the graph each day. Deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    graph = build_movement_graph({"door_activity": 1.0})
    regimes = draw_regime_sequence(rng, n_days)
    start = pd.Timestamp(start_date)

    all_rows = []
    for day_offset, regime in enumerate(regimes):
        date = start + pd.Timedelta(days=day_offset)
        all_rows.extend(
            generate_daily_events(
                rng, date, events_mean, night_ratio, graph, regime
            )
        )
    return pd.DataFrame(all_rows, columns=["date", "time", "sensor_id", "reading"])


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
