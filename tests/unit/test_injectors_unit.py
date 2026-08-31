import numpy as np
import pandas as pd
import pytest

from evaluation.event_injection import (
    COLLECTIVE_INTENSITIES,
    CONTEXTUAL_INTENSITIES,
    POINT_INTENSITIES,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    marginal_diff,
    rare_transition_rate,
    transition_asymmetry,
)
from features import NextEventTransitionExtractor

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
