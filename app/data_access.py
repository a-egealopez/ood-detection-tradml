"""Cached database access helpers for the Streamlit app."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DB_PATH, db_path
from detectors.constants import DEFAULT_CONTAMINATION, DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_SPLIT
from evaluation.event_injection import (
    INTENSITY_PRESETS,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    select_anomaly_dates,
)
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
