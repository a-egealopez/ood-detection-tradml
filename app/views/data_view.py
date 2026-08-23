"""Data step: pick the learning path (2D Playground vs CASAS) and its origin.

The guided workflow starts here: choose between the teaching track (toy 2-D
datasets) and the CASAS track (smart-home event streams), then configure the
data origin (synthetic demo stream vs. real loaded database). The choice is
stored in ``session_state`` and reused by the Features / Detect / Ensemble
steps, so no config lives in this module - only the render logic.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import breadcrumb, clickable_cards, colored_section_header
from theme import (
    FAMILY_BOUNDARY,
    FAMILY_DISTANCE,
    PRIMARY,
    SUCCESS,
    apply_layout,
)

from teaching.datasets import SyntheticDatasetGenerator

DATA_2D = "2D Playground"
DATA_CASAS = "CASAS Smart Home"


def _preview_2d() -> go.Figure:
    """Tiny blob scatter previewing the 2D Playground geometry."""
    X, y = SyntheticDatasetGenerator.generate(
        "blobs",
        n_samples=180,
        contamination=0.12,
        random_state=42,
    )
    normal = y == 0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X[normal, 0],
            y=X[normal, 1],
            mode="markers",
            marker={"size": 4, "color": "rgba(59,130,246,0.55)"},
            hoverinfo="skip",
        )
    )
    if (~normal).sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=X[~normal, 0],
                y=X[~normal, 1],
                mode="markers",
                marker={"size": 5, "color": "#ec4899"},
                hoverinfo="skip",
            )
        )
    apply_layout(fig, None, height=190)
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 10, "b": 10},
        showlegend=False,
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
    )
    return fig


def _preview_casas() -> go.Figure:
    """Tiny event-stream strip previewing the CASAS raw data shape."""
    import pandas as pd

    from features.event_driven_extractors import generate_synthetic_events

    df = generate_synthetic_events(
        n_days=2, pattern="day_night", n_sensors=4, events_per_day=90, seed=7
    )
    ts = pd.to_datetime(df["timestamp"])
    fig = go.Figure()
    for sensor in sorted(df["sensor_id"].unique()):
        idx = df["sensor_id"] == sensor
        fig.add_trace(
            go.Scatter(
                x=ts[idx],
                y=[sensor] * int(idx.sum()),
                mode="markers",
                marker={"size": 4, "color": "rgba(20,184,166,0.6)"},
                hoverinfo="skip",
            )
        )
    apply_layout(fig, None, height=190)
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 10, "b": 10},
        showlegend=False,
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
    )
    return fig


def _render_casas_demo_config() -> None:
    """Demo data panel for the CASAS track (chosen here, used by Features).

    Synthetic: pick a stream size + an optional anomaly scenario. Real: pick how
    many days to read from the loaded database. All keys surface in the Features
    step too.
    """
    origin = st.session_state.get("casas_source", "Synthetic")
    if origin != "Synthetic":
        st.markdown("#### Real CASAS data")
        st.caption(
            "Read from the loaded SQLite database — run the ingestion loader first "
            "(`python src/ingestion/casas_loader.py --source real`)."
        )
        from data_access import list_houses

        try:
            houses = list_houses("real")
        except Exception:  # noqa: BLE001 - DB may not exist until ingestion runs
            houses = []
        if not houses:
            st.warning("No houses found in the real database.")
        else:
            st.caption(f"{len(houses)} house(s) loaded — all of them run the pipeline.")
        st.slider(
            "Days to analyze (from start of dataset)", 3, 30, 10, key="fx_days_real"
        )
        return

    st.markdown("#### Synthetic demo stream")
    st.caption(
        "Tune the size of the stream the Features tutorial will inspect, then pick "
        "the anomaly scenario the detectors should face."
    )

    with st.container(border=True):
        col_days, col_sensors, col_events = st.columns(3, gap="medium")
        with col_days:
            st.slider(
                "Days", 2, 10, int(st.session_state.get("fx_days", 4)), key="fx_days"
            )
        with col_sensors:
            st.slider(
                "Sensors",
                2,
                6,
                int(st.session_state.get("fx_sensors", 3)),
                key="fx_sensors",
            )
        with col_events:
            st.slider(
                "Events / day",
                20,
                200,
                int(st.session_state.get("fx_events_day", 80)),
                step=10,
                key="fx_events_day",
            )

    st.markdown("#### Anomaly scenario")
    st.caption(
        "Optionally inject coherent anomalies (point / contextual / collective) into "
        "the synthetic stream. The injected days are treated as labels, so the ensemble "
        "results can report AUROC. **control** injects nothing — it is the null control "
        "the detectors should treat as noise. Only meaningful on **synthetic** data — "
        "real homes have near-symmetric transitions that a reversal cannot exploit."
    )
    scenario_specs = [
        {
            "id": "control",
            "icon": "🌿",
            "title": "control",
            "description": "Null control — nothing injected, detectors should "
            "stay at chance (~0.5).",
            "color": SUCCESS,
        },
        {
            "id": "point",
            "icon": "💥",
            "title": "point",
            "description": "Night burst: a day that is suddenly very loud — extra "
            "events at 3-4 AM from one sensor.",
            "color": PRIMARY,
        },
        {
            "id": "contextual",
            "icon": "🕐",
            "title": "contextual",
            "description": "Whole routine shifted S hours: an anomalous day happens "
            "at the wrong time of day.",
            "color": FAMILY_DISTANCE,
        },
        {
            "id": "collective",
            "icon": "🔀",
            "title": "collective",
            "description": "Intra-day sensor order partially reversed: the sequence "
            "of rooms is wrong.",
            "color": FAMILY_BOUNDARY,
        },
    ]
    scenario = clickable_cards(scenario_specs, key="fx_scenario")

    if scenario != "control":
        st.segmented_control(
            "Intensity",
            ["low", "medium", "high"],
            default="medium",
            key="fx_scenario_intensity",
            help=(
                "Contextual: hours the routine is shifted (1/3/5h). Collective: "
                "fraction of positions taken from the reversed order (0.25/0.55/1.0)."
            ),
        )
        st.caption(
            "Contextual = routine at the wrong time of day (caught by Z-Score, HMM, "
            "Hawkes). Collective = wrong room order (caught by the next-event Markov "
            "model)."
        )


def render_data_step() -> None:
    """Render the guided-workflow Data step (choose path + origin)."""
    breadcrumb([("Data", True)])
    colored_section_header("1", "Choose your data", PRIMARY)
    st.caption(
        "Choose your learning path: toy **2D** to understand how detectors draw "
        "their boundaries, or **CASAS** event data to detect anomalous days of "
        "activity. Click anywhere on a card to select it."
    )

    data_cards = [
        {
            "id": DATA_2D,
            "icon": "🎮",
            "title": "2D Playground",
            "description": (
                "Toy 2-D datasets (blobs, moons, circles) to watch each "
                "detector carve its own decision boundary."
            ),
            "badge": "Tutorial",
            "color": PRIMARY,
            "figure": _preview_2d(),
        },
        {
            "id": DATA_CASAS,
            "icon": "🏠",
            "title": "CASAS Smart Home",
            "description": (
                "Smart-home event streams: build daily features and score "
                "each day with an ensemble of detectors."
            ),
            "badge": "Real-world",
            "color": FAMILY_BOUNDARY,
            "figure": _preview_casas(),
        },
    ]
    chosen = clickable_cards(data_cards, key="data_source")

    if chosen == DATA_CASAS:
        st.markdown("#### CASAS data origin")
        st.segmented_control(
            "CASAS data",
            ["Synthetic", "Real"],
            key="casas_source",
            default="Synthetic",
        )
        _render_casas_demo_config()
