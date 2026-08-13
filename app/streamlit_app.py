"""Streamlit entry point: guided 3-step workflow (Data -> Features -> Detect)."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components import (
    PATTERN_EXPLANATIONS,
    breadcrumb,
    clickable_cards,
    colored_section_header,
    guided_stepper,
    pattern_preview,
)
from theme import (
    FAMILY_BOUNDARY,
    FAMILY_DISTANCE,
    PRIMARY,
    SUCCESS,
    apply_layout,
    inject_theme,
)
from views import (
    render_casas_view,
    render_documentation_view,
    render_feature_extraction_view,
    render_playground_view,
)

from config import setup_logging
from teaching.datasets import SyntheticDatasetGenerator

st.set_page_config(page_title="Anomaly Detection - CASAS", layout="wide")
inject_theme()
logger = setup_logging()


# ============================================================================
# Top bar: title + Documentation button (opens a dialog, no sidebar)
# ============================================================================
@st.dialog("Documentation & Concepts")
def documentation_dialog() -> None:
    render_documentation_view()


col_title, col_docs = st.columns([5, 1])
with col_title:
    st.title("Anomaly Detection: Health IoT (CASAS)")
    st.caption(
        "Unsupervised ensemble of vectorial and sequential detectors. Follow the "
        "guided workflow: pick data, inspect feature extraction, then run the "
        "detectors."
    )
with col_docs:
    st.write("")
    if st.button("📖 Documentation", key="docs_btn", use_container_width=True):
        documentation_dialog()

st.markdown("---")

# ----------------------------------------------------------------------------
# Guided workflow: Data -> Features -> Detect -> Ensemble
# ----------------------------------------------------------------------------
STEPS = ["Data", "Features", "Detect", "Ensemble"]
DATA_2D = "2D Playground"
DATA_CASAS = "CASAS Smart Home"

if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
if "data_source" not in st.session_state:
    st.session_state.data_source = DATA_2D
if "casas_source" not in st.session_state:
    st.session_state.casas_source = "Synthetic"


def _next() -> None:
    st.session_state.workflow_step = min(
        st.session_state.workflow_step + 1, len(STEPS) - 1
    )


def _back() -> None:
    st.session_state.workflow_step = max(st.session_state.workflow_step - 1, 0)


def _scroll_to_top() -> None:
    """Jump the browser to the top of the window.

    Streamlit keeps the scroll position between reruns, so switching steps (tabs)
    leaves the view down the page. A tiny 0-height component scrolls the parent
    document up, injected only when the active step changes.
    """
    components.html(
        """
        <script>
          const doc = window.parent.document;
          doc.documentElement.scrollTop = 0;
          doc.body.scrollTop = 0;
        </script>
        """,
        height=0,
    )


def _preview_2d() -> go.Figure:
    """Tiny blob scatter previewing the 2D Playground geometry."""
    X, y, _ = SyntheticDatasetGenerator.generate(
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


step = guided_stepper(STEPS, st.session_state.workflow_step, key="workflow_step")

# Scroll the new step into view: the browser would otherwise keep the old
# scroll offset, leaving the tab content below the fold.
if st.session_state.get("_prev_step") != step:
    st.session_state["_prev_step"] = step
    _scroll_to_top()


# ============================================================================
# Step content, boxed so the advance reads as a framed stage
# ============================================================================
def _render_casas_demo_config() -> None:
    """Demo data panel for the CASAS track (chosen here, used by Features).

    Synthetic: pick a temporal pattern + stream size. Real: pick how many days to
    read from the loaded database. All keys surface in the Features step too.
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
        except Exception:
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
        "Tune the size of the stream, then pick the day-of-life pattern the Features tutorial will inspect."
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.slider(
                "Days", 2, 10, int(st.session_state.get("fx_days", 4)), key="fx_days"
            )
        with c2:
            st.slider(
                "Sensors",
                2,
                6,
                int(st.session_state.get("fx_sensors", 3)),
                key="fx_sensors",
            )
        with c3:
            st.slider(
                "Events / day",
                20,
                200,
                int(st.session_state.get("fx_events_day", 80)),
                step=10,
                key="fx_events_day",
            )

    pattern_specs = [
        {
            "id": "regular",
            "icon": "🔄",
            "title": "regular",
            "description": "Steady rhythm, evenly paced events.",
            "color": SUCCESS,
            "figure": pattern_preview("regular"),
        },
        {
            "id": "bursty",
            "icon": "⚡",
            "title": "bursty",
            "description": "Activity clusters with gaps between.",
            "color": FAMILY_DISTANCE,
            "figure": pattern_preview("bursty"),
        },
        {
            "id": "day_night",
            "icon": "🌗",
            "title": "day_night",
            "description": "Active by day, quiet by night.",
            "color": FAMILY_BOUNDARY,
            "figure": pattern_preview("day_night"),
        },
    ]
    pattern = clickable_cards(pattern_specs, key="fx_pattern")
    st.info(PATTERN_EXPLANATIONS[pattern])


def _render_data_step() -> None:
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


def _render_features_step() -> None:
    if st.session_state.data_source == DATA_2D:
        st.markdown("#### 2 · Feature extraction")
        st.caption(
            "Toy 2D datasets need no feature engineering — the raw (x, y) coordinates "
            "are already the feature vector. Press Continue to see how each detector "
            "cuts the space."
        )
    else:
        render_feature_extraction_view()


def _render_detect_step() -> None:
    if st.session_state.data_source == DATA_2D:
        render_playground_view()
    else:
        render_casas_view(str(st.session_state.casas_source).lower(), stage="detect")


def _render_ensemble_step() -> None:
    if st.session_state.data_source == DATA_2D:
        st.markdown("#### 4 · Ensemble")
        st.caption(
            "The 2D playground is about single detectors cutting the space — there "
            "is no ensemble to combine there. Switch to **CASAS Smart Home** to see "
            "the detectors vote on daily activity."
        )
    else:
        render_casas_view(str(st.session_state.casas_source).lower(), stage="ensemble")


with st.container(border=True):
    if step == 0:
        _render_data_step()
    elif step == 1:
        _render_features_step()
    elif step == 2:
        _render_detect_step()
    else:
        _render_ensemble_step()
st.markdown("##")
col_left, col_right = st.columns([1, 1])
with col_left:
    st.button(
        "← Back",
        key="back_btn",
        use_container_width=True,
        disabled=(step == 0),
        on_click=_back,
    )
with col_right:
    if step < len(STEPS) - 1:
        st.button(
            "Continue →",
            key="next_btn",
            use_container_width=True,
            type="primary",
            on_click=_next,
        )
    else:
        st.button(
            "Back to start",
            key="home_btn",
            use_container_width=True,
            on_click=lambda: st.session_state.update(workflow_step=0),
        )

logger.info("Streamlit app executed correctly")
