"""Cached database access helpers for the Streamlit app."""

import logging
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DB_PATH, SYNTHETIC_DATA_DIR, db_path
from detectors.constants import DEFAULT_CONTAMINATION, DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_SPLIT
from evaluation.event_injection import (
    INTENSITY_PRESETS,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    select_anomaly_dates,
)
from ingestion.casas_loader import load_all_houses
from ingestion.markov_generator import generate_house_stream
from ingestion.sqlite_manager import SQLiteDataManager

# Anomaly scenarios the synthetic track can face, in the picker order. "control"
# is the null control (nothing injected); the three real types are injected on
# the raw event stream at the chosen intensity.
INJECTION_SCENARIOS = ("control", "point", "contextual", "collective")

# Fixed split / contamination so the app mirrors the CLI matrix evaluation.
INJECTION_TRAIN_SPLIT = DEFAULT_TRAIN_SPLIT
INJECTION_CONTAMINATION = DEFAULT_CONTAMINATION
INJECTION_SEED = DEFAULT_RANDOM_STATE


def apply_injection(
    df: pd.DataFrame,
    scenario: str,
    intensity_level: str = "medium",
    seed: int = INJECTION_SEED,
) -> tuple[pd.DataFrame, tuple]:
    """Inject the chosen anomaly scenario into an event stream.

    Anomalous days are a contiguous block on the holdout tail (same rule as
    ``matrix_evaluation.prepare_cell``), so detectors train only on clean days.
    Returns ``(injected_df, anomalous_dates)``; ``anomalous_dates`` is empty when
    ``scenario == "control"`` (null control, nothing injected).
    """
    if scenario == "control":
        return df, ()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    dates = sorted(df["date"].unique())

    split_idx = max(1, int(len(dates) * INJECTION_TRAIN_SPLIT))
    holdout_dates = dates[split_idx:]
    if len(holdout_dates) < 1:
        return df.drop(columns=["date"]), ()

    rng = np.random.default_rng(seed)
    n_anomaly = max(1, round(len(holdout_dates) * INJECTION_CONTAMINATION))
    anomaly_dates = select_anomaly_dates(holdout_dates, rng, n_anomaly)

    intensity = INTENSITY_PRESETS[scenario][intensity_level]
    rng_inject = np.random.default_rng(seed)
    if scenario == "point":
        df_inj = inject_point_events(df, rng_inject, intensity, anomaly_dates)
    elif scenario == "contextual":
        df_inj = inject_contextual_events(df, rng_inject, int(intensity), anomaly_dates)
    else:
        df_inj = inject_collective_events(df, rng_inject, intensity, anomaly_dates)
    return df_inj, tuple(sorted(anomaly_dates))


def get_injection_config() -> tuple[str, str]:
    """Returns ``(scenario, intensity)`` from session_state with defaults.

    The scenario/intensity are chosen in the Data step (synthetic track) and
    shared by the Features step and the CASAS Detect step.
    """
    scenario = st.session_state.get("fx_scenario", "control")
    intensity = st.session_state.get("fx_scenario_intensity", "medium")
    return scenario, intensity


# Synthetic house profiles mirroring scripts/generate_test_fixtures.py. Kept here
# so the app can self-provision its data at deploy time (data/ is gitignored).


class _HouseProfile(TypedDict):
    events_mean: int
    night_ratio: float
    n_days: int
    seed: int


_SYNTHETIC_HOUSE_PROFILES: dict[str, _HouseProfile] = {
    "aruba": {"events_mean": 180, "night_ratio": 0.08, "n_days": 90, "seed": 1},
    "cairo": {"events_mean": 90, "night_ratio": 0.05, "n_days": 75, "seed": 2},
    "milan": {"events_mean": 260, "night_ratio": 0.15, "n_days": 85, "seed": 3},
    "tulum": {"events_mean": 130, "night_ratio": 0.10, "n_days": 70, "seed": 4},
}


def ensure_synthetic_db() -> None:
    """Provision the synthetic CASAS database if it is missing.

    ``data/`` is gitignored, so a fresh clone (or a Streamlit Community Cloud
    deploy) has no engine database. This runs at app startup and generates the
    four synthetic houses from the Markov generator, then loads them into SQLite.
    Failures are logged and swallowed so the 2-D teaching track keeps working even
    if synthetic data cannot be built.
    """
    out = db_path("synthetic")
    if out.exists():
        return
    try:
        SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for house_id, profile in _SYNTHETIC_HOUSE_PROFILES.items():
            df = generate_house_stream(house_id=house_id, **profile)
            df.to_csv(SYNTHETIC_DATA_DIR / f"casas_{house_id}_raw.csv", header=False, index=False)
        mgr = SQLiteDataManager(str(out))
        mgr.connect()
        try:
            mgr.create_tables()
            load_all_houses(mgr, source="synthetic")
        finally:
            mgr.close()
    except Exception:  # pragma: no cover - provisioning should never break the UI
        logging.getLogger(__name__).exception("Synthetic data provisioning failed")


@st.cache_data(show_spinner=False)
def list_houses(source: str) -> list[str]:
    """List available house IDs for a data source (real | synthetic)."""
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    try:
        return db.list_houses()
    finally:
        db.close()


@st.cache_data(show_spinner=False)
def load_house_events(source: str, house_id: str) -> pd.DataFrame:
    """Load all events of a single house from the given data source."""
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    try:
        return db.query_house(house_id)
    finally:
        db.close()


@st.cache_data(show_spinner=False)
def load_house_events_injected(
    source: str,
    house_id: str,
    scenario: str = "control",
    intensity_level: str = "medium",
    seed: int = INJECTION_SEED,
) -> tuple[pd.DataFrame, tuple]:
    """Load a house's events and apply the chosen anomaly scenario (synthetic track).

    Returns ``(injected_df, anomalous_dates)``; with ``scenario == "control"`` the
    clean stream comes back untouched and the date tuple is empty.
    """
    df = load_house_events(source, house_id)
    return apply_injection(df, scenario, intensity_level, seed=seed)


@st.cache_data(show_spinner=False)
def load_all_events() -> pd.DataFrame:
    """Load the full real event table (used by the feature-extraction tutorial)."""
    db = SQLiteDataManager(str(DB_PATH))
    db.connect()
    try:
        return db.query_to_dataframe("SELECT * FROM sensor_events ORDER BY timestamp")
    finally:
        db.close()
