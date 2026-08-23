"""Markov-chain event generator for synthetic CASAS-style data.

Purpose
-------
The previous fixture generator drew each event's sensor independently of its
neighbours (``rng.choice``), so the resulting streams had no sequential structure
for the order/transition detectors (next-event Markov, n-gram, ``MarkovSequence``)
to learn. This generator produces the structure on purpose:

1. A **directed movement graph** per house whose transition weights are strongly
   asymmetric and depend on the *hour band* (morning / midday / afternoon / night).
   Asymmetry is mandatory: reversing or shuffling the intra-day order (the
   collective injector) only produces rare transitions when common transitions
   are not symmetric.
2. A **sticky latent day regime** (Markov chain with ``P(stay) ~ 0.75`` and 2-3
   states) that modulates the daily event volume and night ratio, giving the daily
   features day-to-day autocorrelation (structure the HMM/regime detectors can use).

Output
------
Functions return event rows in the exact CSV schema the loader expects
(``date, time, sensor_id, reading``, no header) so ``casas_loader`` keeps working
unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS: list[str] = ["Bedroom", "Kitchen", "LivingRoom", "OutsideDoor"]
MOTION_SENSORS = ("Bedroom", "Kitchen", "LivingRoom")

# Hour bands the movement graph conditions on.
BAND_MORNING, BAND_MIDDAY, BAND_AFTERNOON, BAND_NIGHT = range(4)
BAND_NAMES = ("morning", "midday", "afternoon", "night")

# Probability of picking a uniformly random sensor instead of following the graph.
# Kept small: the graph's structure must dominate or the transition model learns
# noise and the collective injector has nothing rare to exploit.
NOISE = 0.10

# Sticky latent day regimes. A regime scales the daily event count and the night
# ratio. P(stay) high => long runs => autocorrelated daily features.
REGIMES: list[dict] = [
    {"name": "quiet", "event_factor": 0.70, "night_offset": -0.02},
    {"name": "typical", "event_factor": 1.00, "night_offset": 0.00},
    {"name": "active", "event_factor": 1.35, "night_offset": 0.04},
]
REGIME_STAY_PROB = 0.75

# Diurnal activity curve for the day-time hours (8-21): morning peak, midday
# lull, evening peak. This gives the per-hour profile structure, so a circular
# time shift (the contextual injector) moves events between busy and quiet hours
# and the per-hour detectors (Hawkes) can see it. Without the curve the day hours
# are flat and a shift is invisible to hourly models.
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


# Base asymmetric transition weights: ``weights[band][source][target]``, higher =
# more likely. Each source row has one dominant target and several small ones, so
# the reverse/rare transitions the collective injector produces stay improbable.
#
# The between-room structure is a *directed cycle*: Bedroom -> Kitchen ->
# LivingRoom -> Bedroom, with the backward edges (Kitchen->Bedroom,
# LivingRoom->Kitchen, Bedroom->LivingRoom) kept close to zero. This directional
# asymmetry is mandatory: reversing a day's sequence (the collective injector)
# turns the common forward transitions into the rare backward ones, which is what
# makes the order anomaly visible to the transition detectors. Without it, a
# reversed day is indistinguishable from a normal one.
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
    """Return the directed movement graph: ``(band, source) -> {target: weight}``.

    ``profile`` may carry a ``door_activity`` multiplier (default 1.0) that scales
    all weights into/out of the OutsideDoor node, giving each house a slightly
    different daily-routine shape while keeping the mandatory asymmetry.
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
    """Entropy of the generated graph, averaged over (band, source) rows.

    Normalized by ``log2(n_sensors)`` so 1.0 = uniform random transitions. The DoD
    gate requires this below 0.70: the graph must be peaked (structured), not flat.
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
    """Walk the graph one step: sample the next sensor given the previous one and
    the current hour's band, mixed with a small uniform component."""
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
            # Move to one of the other states uniformly.
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
    """Generate one day of events as ``(time_str, sensor_id, reading)`` tuples.

    Timestamps are drawn first (mixing daytime/night-time hours per ``night_ratio``
    and the regime's offset), then a Markov walk on the movement graph assigns each
    event its sensor, conditioned on the event's hour band.
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
    house_id: str,
    events_mean: int,
    night_ratio: float,
    n_days: int,
    seed: int,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate the full event stream for one house as a DataFrame in CSV schema.

    The latent regime is drawn per day, then each day walks the movement graph.
    Deterministic for a given seed. Returns ``(date, time, sensor_id, reading)``.
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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    graph = build_movement_graph({"door_activity": 1.0})
    entropy_ratio = graph_entropy_ratio(graph)
    print(f" Graph entropy ratio: {entropy_ratio:.3f} (gate < 0.70)")
    assert entropy_ratio < 0.70, "movement graph is not peaked enough"

    stream = generate_house_stream(
        house_id="aruba", events_mean=180, night_ratio=0.08, n_days=60, seed=1
    )
    daily_counts = (
        pd.to_datetime(stream["date"])
        .value_counts()
        .sort_index()
        .astype(float)
        .values
    )
    ac1 = first_order_autocorrelation(daily_counts)
    print(f" AC1(n_events): {ac1:.3f} (gate |AC1| >= 0.30)")
    assert abs(ac1) >= 0.30, "daily feature lacks lag-1 autocorrelation"

    n_transitions = 0
    rare = 0
    for date, group in stream.groupby("date"):
        seq = group.sort_values("time")["sensor_id"].tolist()
        for source, target in zip(seq[:-1], seq[1:]):
            n_transitions += 1
            band = hour_to_band(0)  # rough: skip hour conditioning here
            if transition_probabilities(graph, band, source)[target] < 0.10:
                rare += 1
    print(f" Rare transitions (unconditional proxy): {rare} / {n_transitions}")
    print(f" Events: {len(stream)}, days: {stream['date'].nunique()}")
    print(" Validation OK")