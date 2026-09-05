"""Event-level anomaly injectors (point, contextual, collective) for evaluation.

Operate on raw event stream (not feature matrix). Each injector changes its target
marginals and preserves the others:

- inject_point_events: adds a night burst (3-4 AM) from one sensor — a "loud" day
  deviating in aggregate features (n_events, peak_hour, night_activity). No invariants.
- inject_contextual_events: circularly shifts an anomalous day's routine by S hours.
  Preserves total/per-sensor counts and sensor sequence; changes hourly distribution.
- inject_collective_events: reorders intra-day sensor sequence (partial/full reversal).
  Preserves per-sensor AND per-hour counts; changes only transition structure.
  Requires asymmetric transitions (see transition_asymmetry).

All three are intensity-graded (low/medium/high) for monotonic AUROC checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Night window for DoD proxy checks (contextual injector shifts whole days).
DISPLACEMENT_WINDOW = (2, 5)

# Contextual: whole-day shift in hours. Collective: fraction of reversed order.
# Point: extra events as fraction of day's event count.
CONTEXTUAL_INTENSITIES = {"low": 1, "medium": 3, "high": 5}
COLLECTIVE_INTENSITIES = {"low": 0.25, "medium": 0.55, "high": 1.0}
POINT_INTENSITIES = {"low": 0.5, "medium": 1.0, "high": 2.0}

INTENSITY_PRESETS = {
    "point": POINT_INTENSITIES,
    "contextual": CONTEXTUAL_INTENSITIES,
    "collective": COLLECTIVE_INTENSITIES,
}

POINT_BURST_HOUR = (3, 4)  # 3-4 AM: normally quiet


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


def select_anomaly_dates(dates, rng, n_anomaly: int):
    """Pick a contiguous block of ``n_anomaly`` dates (sequential anomalies)."""
    dates = sorted(dates)
    if n_anomaly >= len(dates):
        return set(dates)
    start = int(rng.integers(0, len(dates) - n_anomaly + 1))
    return set(dates[start : start + n_anomaly])


def _event_pair_for(sensor: str, rng) -> tuple[str, float]:
    """A valid (event_type, value) pair for a sensor (door vs motion readings)."""
    if sensor == "OutsideDoor":
        reading = rng.choice(["OPEN", "CLOSE"])
        return reading, 1.0 if reading == "OPEN" else 0.0
    reading = rng.choice(["ON", "OFF"])
    return reading, 1.0 if reading == "ON" else 0.0


def inject_contextual_events(
    df: pd.DataFrame,
    _rng,
    intensity: int,
    anomaly_dates,
) -> pd.DataFrame:
    """Circularly shift each anomalous day's whole routine by ``intensity`` hours.

    Preserves total/per-sensor counts and sensor sequence (only wrap transition
    differs). Anomaly visible only in hourly distribution (night_activity,
    peak_hour), caught by Z-Score/HMM/Hawkes; order detectors stay blind.
    """
    df = _with_date(df)
    shift = pd.Timedelta(hours=int(intensity))
    day = pd.Timedelta(hours=24)

    for date in sorted(anomaly_dates):
        mask = df["date"] == date
        ts = pd.to_datetime(df.loc[mask, "timestamp"])
        time_of_day = ts - ts.dt.normalize()
        new_tod = (time_of_day + shift) % day
        df.loc[mask, "timestamp"] = ts.dt.normalize() + new_tod

    return df.drop(columns=["date"])


def inject_collective_events(
    df: pd.DataFrame,
    rng,
    intensity: float,
    anomaly_dates,
) -> pd.DataFrame:
    """Reorder each anomalous day's sensor sequence, timestamps untouched.

    intensity = fraction of reversed order (1.0 = full reversal). Per-sensor and
    per-hour counts preserved; only transition structure changes (rare when graph
    is asymmetric).
    """
    df = _with_date(df)
    if not anomaly_dates:
        return df.drop(columns=["date"])

    for date in sorted(anomaly_dates):
        day_mask = df["date"] == date
        n = int(day_mask.sum())
        if n < 4:
            continue
        idx = df.index[day_mask]
        sensors = df.loc[idx, "sensor_id"].values
        reversed_order = sensors[::-1]

        new_sensors = sensors.copy()
        # Mirror-pair swaps (i <-> n-1-i) keep multiset intact.
        # Isolated swaps from reversed order would change per-sensor counts.
        n_swapped = max(1, round(intensity * n))
        n_pairs = max(1, min(n // 2, n_swapped // 2))
        half = np.arange(n // 2)
        chosen = rng.choice(half, size=n_pairs, replace=False)
        swap_idx = np.concatenate([chosen, n - 1 - chosen])
        new_sensors[swap_idx] = reversed_order[swap_idx]

        df.loc[idx, "sensor_id"] = new_sensors
        # Keep event_type/value consistent with the (possibly new) sensor type.
        for pos, sensor in zip(idx, new_sensors, strict=True):
            if sensor == "OutsideDoor":
                event_type, value = _event_pair_for(sensor, rng)
                df.loc[pos, "event_type"] = event_type
                df.loc[pos, "value"] = value
    return df.drop(columns=["date"])


def inject_point_events(
    df: pd.DataFrame,
    rng,
    intensity: float,
    anomaly_dates,
    burst_hour: tuple[int, int] = POINT_BURST_HOUR,
) -> pd.DataFrame:
    """Add a night burst of extra events to each anomalous day (a "loud" day).

    intensity = extra events as fraction of day's count (1.0 = double activity).
    Burst at 3-4 AM from single random sensor. Deviates aggregate features
    (n_events, activity_hours, peak_hour, night_activity). No invariants preserved.
    """
    df = _with_date(df)
    if not anomaly_dates:
        return df.drop(columns=["date"])

    start_hour, end_hour = burst_hour
    window_minutes = (end_hour - start_hour) * 60
    pieces = []
    for date, day in df.groupby("date", sort=True):
        if date not in anomaly_dates:
            pieces.append(day)
            continue
        day = day.copy()
        n_extra = max(1, round(len(day) * float(intensity)))
        base = day["timestamp"].iloc[0].normalize() + pd.Timedelta(hours=start_hour)
        offsets = rng.integers(0, window_minutes, size=n_extra)
        sensor = str(rng.choice(day["sensor_id"].unique()))
        event_type, value = _event_pair_for(sensor, rng)
        template = day.iloc[int(rng.integers(0, len(day)))]
        extra = []
        for m in offsets:
            row = template.copy()
            row["timestamp"] = base + pd.Timedelta(minutes=int(m))
            row["sensor_id"] = sensor
            row["event_type"] = event_type
            row["value"] = float(value)
            extra.append(row)
        pieces.append(pd.concat([day, pd.DataFrame(extra)], ignore_index=True))

    df = pd.concat(pieces, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.drop(columns=["date"])


def transition_asymmetry(df: pd.DataFrame, extractor=None) -> float:
    """Mean direction imbalance of pooled transition counts.

    For each unordered pair {a,b}: min(count(a->b), count(b->a)) / max(...).
    1.0 = perfectly symmetric; 0.0 = fully directional. Self-transitions excluded.
    """
    from itertools import pairwise
    from collections import Counter, defaultdict

    if extractor is not None:
        token_col = getattr(extractor, "token_col", "sensor_id")
        from features.daily_aggregates import event_sequence

        sequence = event_sequence(df, token_col)
    else:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sequence = df.sort_values("timestamp")["sensor_id"].astype(str).tolist()

    fwd = Counter(pairwise(sequence))
    pairs = defaultdict(lambda: [0, 0])
    for (a, b), c in fwd.items():
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if a == key[0]:
            pairs[key][0] += c
        else:
            pairs[key][1] += c
    ratios = [min(v[0], v[1]) / max(v[0], v[1]) for v in pairs.values() if max(v) > 0]
    return float(np.mean(ratios)) if ratios else 1.0
