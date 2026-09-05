from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from evaluation.anomaly_injectors import (
    COLLECTIVE_INTENSITIES,
    CONTEXTUAL_INTENSITIES,
    POINT_INTENSITIES,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
)
from features import NextEventTransitionExtractor
from features.daily_aggregates import event_sequence


# DoD proxy functions (moved from source to keep source clean)
def _hourly_counts(df: pd.DataFrame) -> dict[int, int]:
    ts = pd.to_datetime(df["timestamp"])
    return ts.dt.hour.value_counts().sort_index().to_dict()


def marginal_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Compare per-day marginal stats between two dataframes (same day)."""
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
    """Mean direction imbalance of pooled transition counts.

    For each unordered pair {a,b}: min(count(a->b), count(b->a)) / max(...).
    1.0 = perfectly symmetric; 0.0 = fully directional. Self-transitions excluded.
    """
    if extractor is not None:
        token_col = getattr(extractor, "token_col", "sensor_id")
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


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

ANOMALY_DATES_KEYS = {"2024-01-20", "2024-01-21"}


@pytest.fixture
def anomaly_dates():
    return {pd.Timestamp("2024-01-20").date(), pd.Timestamp("2024-01-21").date()}


def test_contextual_keeps_total_and_sensor_counts_changes_hourly(casas_stream, rng, anomaly_dates):
    ctx = inject_contextual_events(
        casas_stream, rng, CONTEXTUAL_INTENSITIES["high"], anomaly_dates
    )
    report = marginal_diff(casas_stream, ctx)
    for date_key, diff in report.items():
        if date_key not in ANOMALY_DATES_KEYS:
            continue
        assert diff["total"] == 0
        assert all(v == 0 for v in diff["sensor"].values())
        assert any(v != 0 for v in diff["hour"].values())


def test_collective_preserves_all_marginals(casas_stream, rng, anomaly_dates):
    for level in COLLECTIVE_INTENSITIES.values():
        coll = inject_collective_events(casas_stream, rng, level, anomaly_dates)
        report = marginal_diff(casas_stream, coll)
        for date_key, diff in report.items():
            if date_key not in ANOMALY_DATES_KEYS:
                continue
            assert diff["total"] == 0
            assert all(v == 0 for v in diff["sensor"].values())
            assert all(v == 0 for v in diff["hour"].values())


def test_point_adds_events_only_on_anomalous_days(casas_stream, rng, anomaly_dates):
    for level in POINT_INTENSITIES.values():
        pt = inject_point_events(casas_stream, rng, level, anomaly_dates)
        report = marginal_diff(casas_stream, pt)
        for date_key, diff in report.items():
            if date_key in ANOMALY_DATES_KEYS:
                assert diff["total"] > 0
                assert any(v > 0 for v in diff["sensor"].values())
                assert any(v != 0 for v in diff["hour"].values())
            else:
                assert diff["total"] == 0
                assert all(v == 0 for v in diff["sensor"].values())


def test_stream_is_asymmetric(casas_stream):
    """Precondition: reversal is only detectable when transitions are directional."""
    assert transition_asymmetry(casas_stream) < 0.85


def test_contextual_proxy_monotonic(casas_stream, rng, anomaly_dates):
    proxies = {}
    for level in CONTEXTUAL_INTENSITIES.values():
        df_inj = inject_contextual_events(casas_stream, rng, level, anomaly_dates)
        l1 = 0
        for d in anomaly_dates:
            hb = (
                pd.to_datetime(
                    casas_stream[casas_stream["timestamp"].dt.date == d]["timestamp"]
                )
                .dt.hour.value_counts()
            )
            ha = (
                pd.to_datetime(
                    df_inj[df_inj["timestamp"].dt.date == d]["timestamp"]
                )
                .dt.hour.value_counts()
            )
            l1 += int(np.abs(hb.sub(ha, fill_value=0)).sum())
        proxies[level] = l1
    lows, meds, highs = (
        proxies[CONTEXTUAL_INTENSITIES["low"]],
        proxies[CONTEXTUAL_INTENSITIES["medium"]],
        proxies[CONTEXTUAL_INTENSITIES["high"]],
    )
    assert lows <= meds <= highs


def test_collective_proxy_monotonic(casas_stream, rng, anomaly_dates):
    extractor = NextEventTransitionExtractor().fit(casas_stream)
    proxies = {}
    for level in COLLECTIVE_INTENSITIES.values():
        df_inj = inject_collective_events(casas_stream, rng, level, anomaly_dates)
        rates = [
            rare_transition_rate(df_inj[df_inj["timestamp"].dt.date == d], extractor)
            for d in anomaly_dates
        ]
        proxies[level] = round(float(np.mean(rates)), 3)
    lows, meds, highs = (
        proxies[COLLECTIVE_INTENSITIES["low"]],
        proxies[COLLECTIVE_INTENSITIES["medium"]],
        proxies[COLLECTIVE_INTENSITIES["high"]],
    )
    assert lows <= meds <= highs


def test_point_proxy_monotonic(casas_stream, rng, anomaly_dates):
    proxies = {}
    for level in POINT_INTENSITIES.values():
        df_inj = inject_point_events(casas_stream, rng, level, anomaly_dates)
        gains = [
            (
                len(df_inj[df_inj["timestamp"].dt.date == d])
                - len(casas_stream[casas_stream["timestamp"].dt.date == d])
            )
            / len(casas_stream[casas_stream["timestamp"].dt.date == d])
            for d in anomaly_dates
        ]
        proxies[level] = round(float(np.mean(gains)), 3)
    lows, meds, highs = (
        proxies[POINT_INTENSITIES["low"]],
        proxies[POINT_INTENSITIES["medium"]],
        proxies[POINT_INTENSITIES["high"]],
    )
    assert lows <= meds <= highs
