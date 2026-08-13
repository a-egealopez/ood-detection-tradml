"""Streamlit entry point: guided 4-step workflow (Data -> Features -> Detect -> Ensemble)."""

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components import guided_stepper
from theme import inject_theme
from views import (
    DATA_2D,
    render_casas_view,
    render_data_step,
    render_documentation_view,
    render_feature_extraction_view,
    render_playground_view,
)

from config import setup_logging

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


step = guided_stepper(STEPS, st.session_state.workflow_step, key="workflow_step")

# Scroll the new step into view: the browser would otherwise keep the old
# scroll offset, leaving the tab content below the fold.
if st.session_state.get("_prev_step") != step:
    st.session_state["_prev_step"] = step
    _scroll_to_top()


# ============================================================================
# Step content, boxed so the advance reads as a framed stage
# ============================================================================
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
        render_data_step()
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
