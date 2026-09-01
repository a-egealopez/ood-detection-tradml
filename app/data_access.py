"""Cached database access helpers for the Streamlit app."""

import logging
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import HOUSES, REAL_DATA_DIR, SYNTHETIC_DATA_DIR, db_path
from detectors.constants import DEFAULT_CONTAMINATION, DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_SPLIT
from evaluation.event_injection import (
    INTENSITY_PRESETS,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    select_anomaly_dates,
)
from features.common import truncate_stream_to_days
from ingestion.casas_loader import load_all_houses
from ingestion.markov_generator import generate_house_stream
from ingestion.sqlite_manager import SQLiteDataManager

# Anomaly scenarios: control (null), point, contextual, collective.
INJECTION_SCENARIOS = ("control", "point", "contextual", "collective")

# Fixed train/contamination to match CLI matrix evaluation.
INJECTION_TRAIN_SPLIT = DEFAULT_TRAIN_SPLIT  # 70% train, 30% holdout
INJECTION_CONTAMINATION = DEFAULT_CONTAMINATION  # Anomaly rate on holdout
INJECTION_SEED = DEFAULT_RANDOM_STATE


def apply_injection(
    df: pd.DataFrame,
    scenario: str,
    intensity_level: str = "medium",
    seed: int = INJECTION_SEED,
) -> tuple[pd.DataFrame, tuple]:
    """Inject anomaly scenario into event stream; returns (df, anomalous_dates)."""
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
    """Get (scenario, intensity) from session_state."""
    scenario = st.session_state.get("fx_scenario", "control")
    intensity = st.session_state.get("fx_scenario_intensity", "medium")
    return scenario, intensity


# Synthetic house profiles (self-provision at deploy time since data/ is gitignored).


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
    """Generate synthetic CASAS DB on first run (idempotent, failures are logged)."""
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
    """List available houses in the source database."""
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    try:
        return db.list_houses()
    finally:
        db.close()


@st.cache_data(show_spinner=False)
def load_house_events(source: str, house_id: str, max_days: int = 0) -> pd.DataFrame:
    """Load house events, optionally truncated to max_days."""
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    try:
        df = db.query_house(house_id)
    finally:
        db.close()
    return truncate_stream_to_days(df, max_days or None)


@st.cache_data(show_spinner=False)
def load_house_events_injected(
    source: str,
    house_id: str,
    scenario: str = "control",
    intensity_level: str = "medium",
    seed: int = INJECTION_SEED,
) -> tuple[pd.DataFrame, tuple]:
    """Load house events and apply anomaly scenario; returns (df, anomalous_dates)."""
    df = load_house_events(source, house_id)
    return apply_injection(df, scenario, intensity_level, seed=seed)


def _query_events_with_limit(
    db: SQLiteDataManager, max_days: int | None
) -> pd.DataFrame:
    """Query sensor_events, optionally limited to first max_days (SQL-level windowing)."""
    if max_days is not None and max_days > 0:
        sql = (
            "SELECT * FROM sensor_events "
            "WHERE substr(timestamp, 1, 10) <= ("
            "  SELECT MAX(d) FROM ("
            "    SELECT DISTINCT substr(timestamp, 1, 10) AS d "
            "    FROM sensor_events ORDER BY d LIMIT ?"
            "  )"
            ") "
            "ORDER BY timestamp"
        )
        return db.query_to_dataframe(sql, params=(float(max_days),))
    return db.query_to_dataframe("SELECT * FROM sensor_events ORDER BY timestamp")


@st.cache_data(show_spinner=False)
def load_all_events(max_days: int = 0, source: str = "real") -> pd.DataFrame:
    """Load event table from the specified source database."""
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    try:
        return _query_events_with_limit(db, max_days or None)
    finally:
        db.close()


def download_casas_data_from_zenodo() -> bool:
    """Download and load real CASAS data from Zenodo. Returns True on success."""
    import tempfile
    import urllib.request
    import zipfile

    zenodo_url = "https://zenodo.org/api/records/17180309/files/new_labeled_data.zip/content"
    house_to_files = {
        "aruba": ["aruba.txt"],
        "cairo": ["cairo.txt"],
        "milan": ["milan.txt"],
        "tulum": ["tulum1.txt", "tulum2.txt"],
    }
    valid_readings = {"ON", "OFF", "OPEN", "CLOSE"}
    event_sensor_prefixes = ("M", "D")

    try:
        with st.spinner("📥 Downloading CASAS data from Zenodo (~150MB)..."):
            zip_path = Path(tempfile.gettempdir()) / "casas_casas_new_labeled_data.zip"
            REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

            urllib.request.urlretrieve(zenodo_url, zip_path)

        with st.spinner("📦 Extracting and processing..."):
            with zipfile.ZipFile(zip_path) as archive:
                for house in HOUSES:
                    house_members = house_to_files[house]
                    rows = []
                    for member in house_members:
                        if member not in archive.namelist():
                            raise FileNotFoundError(f"Missing {member} in archive")
                        with archive.open(member) as fh:
                            lines = fh.read().decode("utf-8").splitlines()
                            for line in lines:
                                fields = line.split()
                                if len(fields) >= 4:
                                    date, time_, sensor_id, reading = fields[:4]
                                    if reading in valid_readings and sensor_id.startswith(event_sensor_prefixes):
                                        rows.append(f"{date},{time_},{sensor_id},{reading}")

                    out_csv = REAL_DATA_DIR / f"casas_{house}_raw.csv"
                    out_csv.write_text("\n".join(rows) + "\n")

        with st.spinner("💾 Loading into database..."):
            mgr = SQLiteDataManager(str(db_path("real")))
            mgr.connect()
            try:
                mgr.create_tables()
                load_all_houses(mgr, source="real")
            finally:
                mgr.close()
            st.cache_data.clear()  # Clear cached list_houses so it reloads

        st.success("✅ CASAS data loaded successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return False
