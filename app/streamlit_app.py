"""Streamlit entry point: guided 4-step workflow (Data -> Features -> Detect -> Ensemble)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components import advance_step, back_step, guided_stepper
from config import setup_logging
from detectors import EnsembleDetector
from detectors.constants import DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE, DEFAULT_RANDOM_STATE
from detectors.factory import build_detectors
from streamlit_config import DETECTOR_DEFAULTS_LIST, DETECTOR_REGISTRY
from teaching.datasets import SyntheticDatasetGenerator
from theme import inject_theme
from views import (
    DATA_2D,
    render_casas_view,
    render_data_step,
    render_feature_extraction_view,
    render_playground_view,
)

st.set_page_config(page_title="Anomaly Detection - CASAS", layout="wide")
inject_theme()
logger = setup_logging()

st.title("Anomaly Detection: ML on Event-Driven Time Series")

#st.markdown("""
#Unsupervised ensemble of **vectorial** and **sequential** detectors.
#
#**Two tracks:**
#- **Scikit-learn Synthetic 2D** — blobs, moons, circles, swiss roll ([sklearn.datasets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets))
#- **CASAS Smart Home** — Aruba, Cairo, Milan, Tulum ([casas.wsu.edu/datasets](https://casas.wsu.edu/datasets/))
#    with synthetic anomaly injection:
#
#  - **Point** — volume bursts (night spikes)
#  - **Contextual** — temporal shifts (routine at wrong hour)
#  - **Collective** — order reversals (same counts, broken transitions)
#""")

st.markdown("---")

# Guided workflow: Data -> Features -> Detect -> Ensemble
STEPS = ["Data", "Features", "Detect", "Ensemble"]

if "data_source" not in st.session_state:
    st.session_state.data_source = DATA_2D
if "casas_source" not in st.session_state:
    st.session_state.casas_source = "Synthetic"

step = guided_stepper(STEPS, key="workflow_step")

# Scroll to top on step change (so the next page opens at the very top). Injected
# as a script because the main area scrolls inside its own container, not the window.
if st.session_state.get("_prev_step") != step:
    st.session_state["_prev_step"] = step
    components.html(
        "<script>"
        "(function(){"
        "var d=window.parent.document;"
        "var roots=[d.querySelector('[data-testid=\"stMain\"]'),"
        "d.querySelector('[data-testid=\"stAppViewContainer\"]'),"
        "d.documentElement,d.body];"
        "function top(){"
        "roots.forEach(function(r){if(r){r.scrollTop=0;r.scrollTo(0,0);}});"
        "window.parent.scrollTo(0,0);"
        "if(d.scrollingElement){d.scrollingElement.scrollTop=0;}"
        "}"
        "top();requestAnimationFrame(top);setTimeout(top,60);setTimeout(top,250);"
        "})();"
        "</script>",
        height=0,
    )


def _render_features_step() -> None:
    if st.session_state.data_source == DATA_2D:
        st.caption("2D datasets need no feature engineering")
    else:
        render_feature_extraction_view()


def _render_detect_step() -> None:
    if st.session_state.data_source == DATA_2D:
        render_playground_view()
    else:
        render_casas_view(str(st.session_state.casas_source).lower(), stage="detect")


def _render_ensemble_step() -> None:
    if st.session_state.data_source == DATA_2D:
        _render_2d_ensemble()
    else:
        render_casas_view(str(st.session_state.casas_source).lower(), stage="ensemble")


def _render_2d_ensemble() -> None:
    """Ensemble results for 2D playground using selected detectors."""
    st.markdown("#### 4 · Ensemble (2D)")

    selected_names = st.session_state.get("det_names_2d", DETECTOR_DEFAULTS_LIST)
    detector_names = [n for n in DETECTOR_REGISTRY if n in selected_names and DETECTOR_REGISTRY[n].category != "Sequential"]
    if not detector_names:
        st.warning("Select at least one detector in the Detect step.")
        return

    # Ensemble mode selector (consistent with CASAS)
    from components import clickable_cards
    from theme import PRIMARY, SUCCESS

    card_specs = [
        {
            "id": "weighted",
            "icon": "⚖️",
            "title": "Weighted (soft)",
            "description": "Blends each detector's continuous score into a weighted average — uses all the information.",
            "badge": "continuous",
            "color": PRIMARY,
        },
        {
            "id": "majority",
            "icon": "🗳️",
            "title": "Majority (hard)",
            "description": "Each detector votes yes/no; a point is flagged only when most detectors agree.",
            "badge": "votes",
            "color": SUCCESS,
        },
    ]
    ensemble_id = clickable_cards(card_specs, key="ens_rule_2d")
    ensemble_mode = "soft" if ensemble_id == "weighted" else "hard"

    # Dataset config (must match playground)
    dataset_key = st.session_state.get("playground_dataset", next(iter(SyntheticDatasetGenerator.DATASETS)))
    n_samples = st.session_state.get("playground_n_samples", 300)
    contamination = st.session_state.get("playground_contamination", 0.15)
    dataset_key_internal = SyntheticDatasetGenerator.DATASETS[dataset_key]

    X, y_true = SyntheticDatasetGenerator.generate(
        dataset_key_internal, n_samples=n_samples, contamination=contamination, random_state=DEFAULT_RANDOM_STATE
    )

    # Build detector params from session state (same as playground cards)
    detector_params = {}
    for name in detector_names:
        spec = DETECTOR_REGISTRY[name]
        detector_params[name] = {}
        for param in spec.params:
            key = f"{name}_{dataset_key_internal}_{n_samples}_{param.arg_name}"
            detector_params[name][param.arg_name] = st.session_state.get(key, param.default)
        if "contamination" in detector_params[name]:
            detector_params[name]["contamination"] = contamination

    detectors = build_detectors(detector_names, detector_params)
    weights = np.ones(len(detectors)) / len(detectors)

    ensemble = EnsembleDetector(detectors=detectors, weights=weights, ensemble_mode=ensemble_mode, ensemble_threshold_percentile=DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE)
    ensemble.fit(X)

    anomalies, scores, details = ensemble.predict(X)
    details["y_true"] = y_true

    # Metrics
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    n_true_anomalies = int(y_true.sum())
    n_detected = int(anomalies.sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble AUROC", f"{auroc:.3f}")
    col2.metric("Anomalies Detected", f"{n_detected} / {n_true_anomalies}")
    col3.metric("Recall", f"{n_detected / n_true_anomalies * 100:.1f}%" if n_true_anomalies > 0 else "N/A")

    # Timeline
    fig_timeline = go.Figure(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode="markers",
        marker={"color": anomalies, "colorscale": [[0, "#6366f1"], [1, "#ef4444"]], "size": 8, "line": {"width": 0.5, "color": "rgba(255,255,255,0.35)"}},
        hovertemplate="Index: %{x}<br>Score: %{y:.3f}<extra></extra>",
        name="Ensemble Score",
    ))
    if ensemble.threshold is not None:
        fig_timeline.add_hline(y=ensemble.threshold, line={"color": "#ef4444", "width": 1.5, "dash": "dash"}, annotation_text="threshold (p90)")
    fig_timeline.update_layout(title="Ensemble Scores", xaxis_title="Sample Index", yaxis_title="Score", yaxis_range=[0, 1], height=350, showlegend=False)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Score distribution
    fig_hist = go.Figure(go.Histogram(x=scores, nbinsx=30, marker_color="#6366f1", name="Score Distribution"))
    if ensemble.threshold is not None:
        fig_hist.add_vline(x=ensemble.threshold, line={"color": "#ef4444", "width": 1.5, "dash": "dash"}, annotation_text="threshold")
    fig_hist.update_layout(title="Score Distribution", xaxis_title="Score", yaxis_title="Count", height=250)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Per-detector AUROC
    score_cols = [c for c in details.columns if c.endswith("_score") and c != "ensemble_score"]
    if score_cols:
        st.markdown("**Per-Detector AUROC**")
        auroc_data = []
        for col in score_cols:
            det_name = col.replace("_score", "")
            try:
                det_auroc = roc_auc_score(y_true, details[col])
            except ValueError:
                det_auroc = 0.5
            auroc_data.append({"Detector": det_name, "AUROC": f"{det_auroc:.3f}"})
        st.dataframe(pd.DataFrame(auroc_data), hide_index=True, use_container_width=True)

    # Flagged points
    flagged = details[details["is_anomaly"] == 1].sort_values("ensemble_score", ascending=False)
    st.markdown("**Flagged as Anomalous**")
    st.dataframe(flagged[["ensemble_score", "is_anomaly", *score_cols]], use_container_width=True)


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
        on_click=back_step,
        args=("workflow_step",),
    )
with col_right:
    if step < len(STEPS) - 1:
        st.button(
            "Continue →",
            key="next_btn",
            use_container_width=True,
            type="primary",
            on_click=advance_step,
            args=("workflow_step", len(STEPS)),
        )
    else:
        st.button(
            "Back to start",
            key="home_btn",
            use_container_width=True,
            on_click=lambda: st.session_state.update(workflow_step_current=0, workflow_step_completed=0),
        )

logger.info("Streamlit app executed correctly")
