"""Event-level anomaly injectors (point + contextual + collective) for the evaluation.

These operate on the *raw event stream* (not the extracted feature matrix), so the
anomaly is introduced at the level where the sequential/order detectors live, and
the features are re-extracted afterwards. Each injector changes the marginal
statistics it is *supposed* to change and preserves the ones the other family of
detectors would otherwise catch "by accident" (see ``docs/anomaly_taxonomy.md``):

- ``inject_point_events``: *adds* a burst of extra events from one sensor at a fixed
  unusual hour (3-4 AM) on an otherwise normal day — a "loud" day whose aggregate
  features deviate. Changes total, per-sensor AND hourly counts by construction
  (a volume anomaly has no invariants to preserve). Catches: distance / vectorial
  family, and any model watching counts. Clean days are untouched.
- ``inject_contextual_events``: *circularly shifts* an anomalous day's whole routine
  by ``S`` hours (the resident behaved normally, just at the wrong time of day). Keeps
  total, per-sensor counts and the sensor sequence; changes only the hourly
  distribution — which is the context signal. Catches: Z-Score(night_activity),
  predictive HMM, Hawkes (per-hour). Blind: next-event / Markov (order untouched).
- ``inject_collective_events``: *reorders* a day's intra-day sensor sequence
  (partial shuffle -> full reversal) while keeping every timestamp in place. Keeps
  per-sensor AND per-hour counts identical; changes only the transition structure.
  Catches: next-event / MarkovSequence / n-gram.  Blind: distance, HMM, Hawkes.
  Requires *asymmetric* transitions to be visible (see ``transition_asymmetry``):
  on symmetric data (typical real homes) a reversal produces no rare transitions.

All three are intensity-graded (low / medium / high) so the evaluation can check
that a detector's AUROC rises monotonically with intensity.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# Night window where displaced events are parked (02:00-05:00). Only used by the
# DoD proxy checks; the contextual injector itself shifts whole days circularly.
DISPLACEMENT_WINDOW = (2, 5)

# Contextual: whole-day circular shift in hours (the routine happened S hours
# later than usual). Collective: fraction of positions taken from the reversed
# order (1.0 = full intra-day reversal). Point: extra night events as a fraction
# of the day's own event count (1.0 = the burst doubles the day's activity).
CONTEXTUAL_INTENSITIES = {"low": 1, "medium": 3, "high": 5}
COLLECTIVE_INTENSITIES = {"low": 0.25, "medium": 0.55, "high": 1.0}
POINT_INTENSITIES = {"low": 0.5, "medium": 1.0, "high": 2.0}

# Hour band where the point-anomaly burst is parked (3-4 AM: normally quiet).
POINT_BURST_HOUR = (3, 4)


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
    rng,
    intensity: int,
    anomaly_dates,
) -> pd.DataFrame:
    """Circularly shift each anomalous day's whole routine by ``intensity`` hours.

    Every event's clock time moves ``intensity`` hours later, wrapping around
    midnight within the same calendar date. Total and per-sensor daily counts are
    preserved exactly and the sensor *sequence is preserved* (the chronologically
    sorted order rotates, so only the single wrap transition differs). The anomaly
    is therefore only visible in the hourly distribution / context features
    (night_activity, peak_hour, ...), which is the signal the context-sensitive
    detectors (HMM, Hawkes, Z-Score) are meant to catch; order detectors stay blind.
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

    ``intensity`` = fraction of positions taken from the reversed order; 1.0 = the
    full intra-day sequence is reversed. Because the reversed order is a permutation
    of the day's own sensors and timestamps are never moved, per-sensor AND
    per-hour daily counts are preserved exactly; only the transition structure
    changes (rare when the movement graph is asymmetric).
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
        # Genuine transpositions (i <-> n-1-i) selected in mirror-closed pairs.
        # Swapping the value at i with the value at its mirror keeps the day's
        # multiset intact: assigning isolated positions from the reversed order
        # (without moving the displaced value out) would change the per-sensor
        # counts and leak a feature anomaly the collective injector must not
        # produce.
        n_swapped = max(1, int(round(intensity * n)))
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

    ``intensity`` = extra events as a fraction of the day's own event count
    (1.0 doubles the day's activity). The burst is parked at a fixed, normally
    quiet hour band (3-4 AM) and fired from a single random sensor, so the day's
    aggregate features (``n_events``, ``activity_hours``, ``peak_hour``,
    ``night_activity``) deviate from the training distribution — the classic
    *point* anomaly on the daily features. Clean days and the rest of the stream
    are untouched. A point anomaly is allowed to change any marginal statistic:
    the burst *is* the anomaly, so there are no invariants to preserve.
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
        n_extra = max(1, int(round(len(day) * float(intensity))))
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


# ---------------------------------------------------------------------------
# DoD helpers: marginal preservation and intensity-monotonicity proxies
# ---------------------------------------------------------------------------


def _hourly_counts(df: pd.DataFrame) -> dict[int, int]:
    ts = pd.to_datetime(df["timestamp"])
    return ts.dt.hour.value_counts().sort_index().to_dict()


def marginal_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Compare per-day marginal stats between two dataframes (same day).

    Returns per-day dicts of differences in total events, per-sensor counts and
    per-hour counts. ``df_before``/``df_after`` may span several days; each day is
    compared independently.
    """
    before = _with_date(df_before)
    after = _with_date(df_after)
    report = {}
    for date in sorted(set(before["date"])):
        b = before[before["date"] == date]
        a = after[after["date"] == date]
        report[str(date)] = {
            "total": int(len(a) - len(b)),
            "sensor": {
                k: int(a["sensor_id"].value_counts().get(k, 0) - v)
                for k, v in b["sensor_id"].value_counts().items()
            },
            "hour": {
                k: int(_hourly_counts(a).get(k, 0) - _hourly_counts(b).get(k, 0))
                for k in set(_hourly_counts(b)) | set(_hourly_counts(a))
            },
        }
    return report


def rare_transition_rate(
    df_anomalous: pd.DataFrame, extractor
) -> float:
    """Fraction of an anomalous day's transitions the model marks as rare."""
    logprobs = extractor._transition_logprobs(df_anomalous)
    if not logprobs:
        return 0.0
    return sum(1.0 for lp in logprobs if lp < extractor.rare_threshold_) / len(logprobs)


def transition_asymmetry(df: pd.DataFrame, extractor=None) -> float:
    """Mean direction imbalance of the pooled transition counts.

    For every unordered pair of *distinct* sensors ``{a, b}`` we take
    ``min(count(a->b), count(b->a)) / max(...)`` and average. 1.0 = perfectly
    symmetric (a home where you move to and from each room equally often); 0.0 =
    fully directional. Self-transitions (a->a) are excluded — they carry no
    directionality information. The collective (order-reversal) injector is only
    detectable by first-order transition models when this ratio is clearly below
    1.0 — the reversed transitions must be *rare*, which requires the forward
    direction to dominate. Real home data is usually near-symmetric, so the
    reversal injector has nothing to exploit there by construction.
    """
    if extractor is not None:
        sequence = extractor._sequence(df)
    else:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sequence = df.sort_values("timestamp")["sensor_id"].astype(str).tolist()

    from collections import Counter, defaultdict

    fwd = Counter(itertools.pairwise(sequence))
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


if __name__ == "__main__":
    from features import NextEventTransitionExtractor
    from ingestion.markov_generator import (
        build_movement_graph,
        generate_daily_events,
    )

    rng = np.random.default_rng(11)
    graph = build_movement_graph({"door_activity": 1.0})

    # Build a normal 40-day stream (regime-driven) for the rare-transition model.
    days = []
    for d in range(40):
        rows = generate_daily_events(
            rng, pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), 180, 0.08,
            graph, {"name": "typical", "event_factor": 1.0, "night_offset": 0.0},
        )
        days.append(rows)
    flat = [row for day in days for row in day]
    df = pd.DataFrame(flat, columns=["date", "time", "sensor_id", "reading"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_type"] = ["ON"] * len(df)
    df["value"] = [1.0] * len(df)
    df = df[["timestamp", "sensor_id", "event_type", "value"]]

    rng = np.random.default_rng(11)
    anomaly_dates = {pd.Timestamp("2024-01-20").date(), pd.Timestamp("2024-01-21").date()}

    # --- Contextual: marginals preserved except the hourly distribution ---------
    ctx = inject_contextual_events(df, rng, CONTEXTUAL_INTENSITIES["high"], anomaly_dates)
    ctx_report = marginal_diff(df, ctx)
    ctx_anomaly_days = {str(d) for d in anomaly_dates}
    for date_key, diff in ctx_report.items():
        if date_key not in ctx_anomaly_days:
            continue
        assert diff["total"] == 0, f"{date_key}: total changed ({diff['total']})"
        assert all(v == 0 for v in diff["sensor"].values()), f"{date_key}: sensor counts changed"
        assert any(v != 0 for v in diff["hour"].values()), f"{date_key}: hourly should change"
    print(" Contextual margins OK (total + per-sensor constant, hourly changed)")

    # --- Collective: marginals fully preserved, transitions changed -----------
    for name, level in COLLECTIVE_INTENSITIES.items():
        coll = inject_collective_events(df, rng, level, anomaly_dates)
        coll_report = marginal_diff(df, coll)
        for date_key, diff in coll_report.items():
            if date_key not in ctx_anomaly_days:
                continue
            assert diff["total"] == 0, f"{date_key}: total changed"
            assert all(v == 0 for v in diff["sensor"].values()), (
                f"{date_key}: sensor counts changed ({diff['sensor']})"
            )
            assert all(v == 0 for v in diff["hour"].values()), (
                f"{date_key}: hourly counts changed"
            )
    print(" Collective margins OK (total + per-sensor + per-hour constant, all intensities)")

    # --- Point: the burst only touches anomalous days; clean days untouched ----
    for name, level in POINT_INTENSITIES.items():
        pt = inject_point_events(df, rng, level, anomaly_dates)
        pt_report = marginal_diff(df, pt)
        for date_key, diff in pt_report.items():
            if date_key in ctx_anomaly_days:
                assert diff["total"] > 0, f"{date_key}: burst should add events"
                assert any(v > 0 for v in diff["sensor"].values()), (
                    f"{date_key}: burst should add events for its sensor"
                )
                assert any(v != 0 for v in diff["hour"].values()), (
                    f"{date_key}: burst should change the hourly distribution"
                )
            else:
                assert diff["total"] == 0 and all(v == 0 for v in diff["sensor"].values()), (
                    f"{date_key}: clean day changed by point injector"
                )
    print(" Point margins OK (burst adds events, clean days untouched, all intensities)")

    # --- Asymmetry precondition ---------------------------------------------
    # The collective injector only produces *rare* transitions when the movement
    # graph is directional. Check the generated stream is asymmetric enough, or
    # the reversal proxy below would pass vacuously (as it does on symmetric real
    # data, where reversal is undetectable by construction).
    asym = transition_asymmetry(df)
    print(f" Transition asymmetry (1.0 = symmetric): {asym:.3f} (gate < 0.85)")
    assert asym < 0.85, "collective injection needs asymmetric transitions"

    # --- Intensity monotonicity proxies ---------------------------------------
    extractor = NextEventTransitionExtractor().fit(df)

    ctx_proxies = {}
    for name, level in CONTEXTUAL_INTENSITIES.items():
        df_inj = _with_date(inject_contextual_events(df, rng, level, anomaly_dates))
        b = _with_date(df)
        l1 = 0
        for d in sorted(anomaly_dates):
            hb = pd.to_datetime(b.loc[b["date"] == d, "timestamp"]).dt.hour.value_counts()
            ha = pd.to_datetime(df_inj.loc[df_inj["date"] == d, "timestamp"]).dt.hour.value_counts()
            l1 += int(np.abs(hb.sub(ha, fill_value=0)).sum())
        ctx_proxies[name] = l1
    print(" Contextual proxy (hourly-distribution L1 shift):", ctx_proxies)
    assert ctx_proxies["low"] <= ctx_proxies["medium"] <= ctx_proxies["high"]

    coll_proxies = {}
    for name, level in COLLECTIVE_INTENSITIES.items():
        df_inj = _with_date(inject_collective_events(df, rng, level, anomaly_dates))
        rates = [
            rare_transition_rate(df_inj[df_inj["date"] == d], extractor)
            for d in sorted(anomaly_dates)
        ]
        coll_proxies[name] = round(float(np.mean(rates)), 3)
    print(" Collective proxy (rare-transition rate):", coll_proxies)
    assert coll_proxies["low"] <= coll_proxies["medium"] <= coll_proxies["high"]

    point_proxies = {}
    for name, level in POINT_INTENSITIES.items():
        b = _with_date(df)
        df_inj = _with_date(inject_point_events(df, rng, level, anomaly_dates))
        gains = [
            (len(df_inj[df_inj["date"] == d]) - len(b[b["date"] == d])) / len(b[b["date"] == d])
            for d in sorted(anomaly_dates)
        ]
        point_proxies[name] = round(float(np.mean(gains)), 3)
    print(" Point proxy (relative event gain):", point_proxies)
    assert point_proxies["low"] <= point_proxies["medium"] <= point_proxies["high"]

    print(" Validation OK")