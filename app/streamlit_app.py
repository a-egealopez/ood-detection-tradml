"""Streamlit entry point: top-level navigation and view dispatch."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from views import (  # noqa: E402
    render_casas_view,
    render_documentation_view,
    render_feature_extraction_view,
    render_teaching_view,
)

from config import setup_logging  # noqa: E402

st.set_page_config(page_title="Anomaly Detection - CASAS", layout="wide")
logger = setup_logging()

st.title("Anomaly Detection: Health IoT (CASAS)")
st.caption(
    "Unsupervised ensemble of vectorial and sequential detectors. Each house is trained "
    "independently using synthetic anomaly injection for evaluation."
)

MODE_TEACHING = "Teaching: Synthetic Datasets"
MODE_SYNTHETIC = "Synthetic CASAS Data"
MODE_REAL = "Real CASAS Data"
MODE_DOCUMENTATION = "Documentation"

SUB_MODE = ["Feature Extraction Tutorial", "Anomaly Detection Pipeline"]

mode = st.radio(
    "Choose data source:",
    [MODE_TEACHING, MODE_SYNTHETIC, MODE_REAL, MODE_DOCUMENTATION],
    horizontal=True,
    index=1,
)

if mode == MODE_TEACHING:
    render_teaching_view()

elif mode in (MODE_SYNTHETIC, MODE_REAL):
    source = "synthetic" if mode == MODE_SYNTHETIC else "real"
    st.markdown("---")
    sub_mode = st.radio(
        "View mode", SUB_MODE, horizontal=True, index=1, key=f"view_mode_{source}"
    )
    if sub_mode == SUB_MODE[0]:
        render_feature_extraction_view()
    else:
        render_casas_view(source)

else:
    st.markdown("---")
    render_documentation_view()

logger.info("Streamlit app executed correctly")
