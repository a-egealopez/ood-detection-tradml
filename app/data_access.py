"""Cached database access helpers for the Streamlit app."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DB_PATH, db_path  # noqa: E402
from ingestion.sqlite_manager import SQLiteDataManager  # noqa: E402


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
def load_all_events() -> pd.DataFrame:
    """Load the full real event table (used by the feature-extraction tutorial)."""
    db = SQLiteDataManager(str(DB_PATH))
    db.connect()
    try:
        return db.query_to_dataframe("SELECT * FROM sensor_events ORDER BY timestamp")
    finally:
        db.close()
