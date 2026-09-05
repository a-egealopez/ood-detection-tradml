"""Shared daily-aggregation helpers for the feature extractors."""

import numpy as np
import pandas as pd

from config import EPSILON


class FeatureScaler:
    """Z-score standardizer (fit on train only) shared by the pipeline and CLI."""

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("You must call fit() before transform()")
        X = np.asarray(X, dtype=float)
        return (X - self.mu) / (self.sigma + EPSILON)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def entropy(counts: np.ndarray, base: float = np.e) -> float:
    """Shannon entropy of a count histogram in log ``base`` (0.0 if empty)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    log_base = float(np.log(base)) if base != np.e else 1.0
    return float(-np.sum(probs * np.log(probs + EPSILON)) / log_base)


# Night window (22:00-08:00, wraps past midnight).
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 8


def daily_aggregates(
    group: pd.DataFrame,
    *,
    base: float = np.e,
    include_peak_hour: bool = False,
    include_frequency_std: bool = False,
) -> dict:
    """Reduce one day's events into shared daily statistics.

    Shared by the pipeline and didactic extractors. ``group`` must have parsed
    ``timestamp`` and derived ``hour`` columns; the flags add pipeline-only features.
    """
    n_events = len(group)
    if n_events == 0:
        return {
            "n_events": 0,
            "n_sensors": 0,
            "activity_hours": 0,
            "avg_gap_minutes": 0.0,
            "peak_hour": 0,
            "night_activity": 0.0,
            "event_frequency_std": 0.0,
            "entropy_hourly": 0.0,
            "entropy_sensor": 0.0,
        }

    n_sensors = group["sensor_id"].nunique()
    activity_hours = group["hour"].nunique()

    ts_sorted = group["timestamp"].sort_values()
    gaps = ts_sorted.diff().dropna().dt.total_seconds() / 60.0
    avg_gap_minutes = float(gaps.mean()) if len(gaps) > 0 else 0.0

    hour_mode = group["hour"].mode()
    peak_hour = int(hour_mode.iloc[0]) if len(hour_mode) > 0 else 12

    night_mask = (group["hour"] < NIGHT_END_HOUR) | (
        group["hour"] >= NIGHT_START_HOUR
    )
    night_activity = float(night_mask.sum()) / n_events

    events_per_sensor = group.groupby("sensor_id").size()
    event_frequency_std = (
        float(np.asarray(events_per_sensor).std()) if n_sensors > 1 else 0.0
    )

    hourly_counts = (
        group.groupby("hour").size().reindex(range(24), fill_value=0).values
    )

    result = {
        "n_events": int(n_events),
        "n_sensors": int(n_sensors),
        "activity_hours": int(activity_hours),
        "avg_gap_minutes": avg_gap_minutes,
        "night_activity": night_activity,
        "entropy_hourly": entropy(np.asarray(hourly_counts), base=base),
        "entropy_sensor": entropy(np.asarray(events_per_sensor), base=base),
    }
    if include_peak_hour:
        result["peak_hour"] = peak_hour
    if include_frequency_std:
        result["event_frequency_std"] = event_frequency_std
    return result


def event_sequence(df: pd.DataFrame, token_col: str = "sensor_id") -> list[str]:  # noqa: S107
    """Order a stream's tokens by timestamp (shared by the order-based extractors)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")[token_col].astype(str).tolist()


def truncate_stream_to_days(df: pd.DataFrame, max_days: int | None) -> pd.DataFrame:
    """Keep only the first ``max_days`` chronological days of a raw event stream.

    Used by the CLIs to cap the evaluation window on long real datasets (the full
    WSU homes span up to ~235 days) without re-extracting features from the whole
    stream every seed. The temporal 70/30 split then applies within the truncated
    window, preserving the train-prefix / holdout-tail design.
    """
    if max_days is None or max_days <= 0:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dates = sorted(df["timestamp"].dt.date.unique())
    if len(dates) <= max_days:
        return df
    keep = set(dates[:max_days])
    mask = df["timestamp"].dt.date.isin(keep)
    return df.loc[mask, :].reset_index(drop=True)


def extract_by_date(
    df: pd.DataFrame, feature_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each day through ``feature_fn`` after parsing timestamp/date/hour.

    Returns ``(X, dates)``, one feature row per day.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    rows, dates = [], []
    for date, group in df.groupby("date"):
        rows.append(feature_fn(group))
        dates.append(date)
    return np.array(rows), np.array(dates)
