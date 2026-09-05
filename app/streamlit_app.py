"""Streamlit entry point: guided 4-step workflow (Data -> Features -> Detect -> Ensemble)."""

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components import advance_step, back_step, clickable_cards, guided_stepper
from config import setup_logging
from data_access import ensure_synthetic_db
from detectors import EnsembleDetector
from detectors.constants import DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE, DEFAULT_RANDOM_STATE
from detectors.factory import build_detectors
from faq import render_faq_html
from streamlit_config import DETECTOR_DEFAULTS_LIST, DETECTOR_REGISTRY
from teaching.synthetic_2d_datasets import SyntheticDatasetGenerator
from theme import PRIMARY, SUCCESS, inject_theme
from views import (
    DATA_2D,
    render_casas_view,
    render_data_step,
    render_extractor_inspector,
    render_playground_view,
)

st.set_page_config(page_title="Anomaly Detection - CASAS", layout="wide")
inject_theme()
logger = setup_logging()

# Provision synthetic CASAS data on first run (data/ is gitignored, so a fresh
# deploy has no engine database). Best-effort: failures never break the UI.
ensure_synthetic_db()

# Header bar (sticky) above everything: FyQ button top-right.
# Injected as HTML because a fixed overlay requires custom CSS we can't express
# with native widgets. The button toggles the FAQ modal via the DOM click.
st.session_state.setdefault("faq_open", False)

st.html(
    f"""
    <div class="app-header">
      <button class="fyq-btn" id="fyq-toggle"
              data-faq-open="{'1' if st.session_state.faq_open else '0'}">
        FyQ
      </button>
    </div>

    <style>
      .app-header {{
        position: sticky; top: 0; z-index: 1000;
        display: flex; align-items: center; justify-content: flex-end;
        padding: 8px 4px; margin-bottom: 12px;
        background: var(--bg-canvas);
      }}
      .fyq-btn {{
        flex: 0 0 auto; cursor: pointer; font-weight: 700; font-size: 14px;
        padding: 8px 18px; border-radius: 999px; border: 1px solid var(--border);
        background: var(--bg-surface); color: var(--text-primary);
      }}
      .fyq-btn:hover {{ border-color: var(--action-primary); color: var(--action-primary); }}

      /* FAQ modal overlay */
      .fq-modal-backdrop {{
        position: fixed; inset: 0; z-index: 2000;
        background: rgba(10, 12, 20, 0.62);   /* semi-transparent page overlay */
        backdrop-filter: blur(2px);
        display: flex; align-items: center; justify-content: center;
        padding: 24px;
      }}
      .fq-modal {{
        background: var(--bg-surface); color: var(--text-primary);
        border: 1px solid var(--border); border-radius: 12px;
        width: min(760px, 100%); max-height: 86vh; overflow-y: auto;
        padding: 26px 30px; position: relative;
        box-shadow: 0 24px 60px rgba(0,0,0,0.5);
      }}
      .fq-modal h3 {{ margin-top: 0; }}
      .fq-group-title {{
        margin: 20px 0 6px; padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
        font-size: 13px; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; color: var(--text-muted);
      }}
      .fq-group-title:first-child {{ margin-top: 0; }}
      .fq-list {{ display: flex; flex-direction: column; gap: 14px; }}
      .fq-card {{
        border: 1px solid var(--border); border-left: 4px solid var(--action-primary);
        border-radius: 8px; background: rgba(255,255,255,0.03);
        padding: 12px 16px;
      }}
      .fq-q {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
        font-weight: 700; font-size: 15px;
      }}
      .fq-num {{
        flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center;
        width: 22px; height: 22px; border-radius: 6px; font-size: 13px;
        background: var(--action-primary); color: #fff;
      }}
      .fq-answer {{
        line-height: 1.55; font-size: 14px; color: var(--text-primary); margin-bottom: 8px;
      }}
      .fq-answer code {{
        background: rgba(255,255,255,0.07); padding: 1px 6px; border-radius: 5px;
        font-size: 0.9em;
      }}
      .fq-refs {{
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
        font-size: 12px; color: var(--text-muted);
      }}
      .fq-refs-label {{ color: var(--text-muted); }}
      .fq-ref {{
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 24px; padding: 2px 7px; border-radius: 999px;
        background: var(--bg-surface); border: 1px solid var(--border);
        color: var(--action-primary); text-decoration: none; font-weight: 700;
      }}
      .fq-ref:hover {{ border-color: var(--action-primary); background: var(--action-primary); color: #fff; }}
      .fq-close {{
        position: absolute; top: 14px; right: 16px; cursor: pointer;
        background: transparent; border: none; color: var(--text-muted);
        font-size: 22px; line-height: 1;
      }}
      .fq-close:hover {{ color: var(--text-primary); }}
    </style>
    """
)

st.title("Anomaly Detection: ML on Event-Driven Time Series")

# The modal overlay itself (always injected; the header button toggles it in JS).
st.html(
    f"""
    <div id="fq-modal-wrap" class="fq-modal-backdrop"
         style="display:{'flex' if st.session_state.faq_open else 'none'};">
      <div class="fq-modal">
        <button class="fq-close" data-fq-dismiss aria-label="Close">✕</button>
        <h3>Frequently Asked Questions</h3>
        {render_faq_html()}
      </div>
    </div>
    <script>
    (function () {{
      function wrap() {{ return document.getElementById('fq-modal-wrap'); }}
      function setOpen(open) {{
        var toggle = document.getElementById('fyq-toggle');
        if (toggle) {{ toggle.setAttribute('data-faq-open', open ? '1' : '0'); }}
        wrap().style.display = open ? 'flex' : 'none';
      }}
      var toggle = document.getElementById('fyq-toggle');
      if (toggle) {{
        toggle.addEventListener('click', function () {{
          setOpen(toggle.getAttribute('data-faq-open') !== '1');
        }});
      }}
      wrap().addEventListener('click', function (e) {{
        if (e.target === wrap() || e.target.hasAttribute('data-fq-dismiss')) {{
          setOpen(false);
        }}
      }});
    }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)

st.markdown("---")

# Guided workflow: Data -> Features -> Detect -> Ensemble
STEPS = ["Data", "Features", "Detect", "Ensemble"]

if "data_source" not in st.session_state:
    st.session_state.data_source = DATA_2D
if "casas_source" not in st.session_state:
    st.session_state.casas_source = "Synthetic"

step = guided_stepper(STEPS, key="workflow_step")

# Scroll to top on step change (so the next page opens at the very top). Injected
# as an inline script because the main area scrolls inside its own container, not
# the window. Replaces the deprecated st.components.v1.html with st.html.
if st.session_state.get("_prev_step") != step:
    st.session_state["_prev_step"] = step
    st.html(
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
        unsafe_allow_javascript=True,
    )


def _render_features_step() -> None:
    if st.session_state.data_source == DATA_2D:
        st.caption("2D datasets need no feature engineering")
    else:
        render_extractor_inspector()


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


@st.cache_data(show_spinner=False, max_entries=128)
def _ensemble_2d_results(
    detector_names: tuple[str, ...],
    detector_params: dict,
    dataset_key_internal: str,
    n_samples: int,
    contamination: float,
    ensemble_mode: Literal["soft", "hard"],
) -> dict:
    """Build, fit and score the 2D ensemble once per config signature (cached).

    Returns only picklable results (scores, threshold, AUROCs, flagged rows) so
    the expensive detector fits run once per parameter set instead of every
    rerun. Figures are drawn in the view from these cached numbers.
    """
    X, y_true = SyntheticDatasetGenerator.generate(
        dataset_key_internal,
        n_samples=n_samples,
        contamination=contamination,
        random_state=DEFAULT_RANDOM_STATE,
    )
    detectors = build_detectors(list(detector_names), detector_params)
    weights = np.ones(len(detectors)) / len(detectors)

    ensemble = EnsembleDetector(
        detectors=detectors,
        weights=weights,
        ensemble_mode=ensemble_mode,
        ensemble_threshold_percentile=DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
    )
    ensemble.fit(X)

    anomalies, scores, details = ensemble.predict(X)
    details["y_true"] = y_true

    n_true = int(y_true.sum())
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    score_cols = [c for c in details.columns if c.endswith("_score") and c != "ensemble_score"]
    per_det_auroc: dict[str, float] = {}
    for col in score_cols:
        try:
            per_det_auroc[col] = roc_auc_score(y_true, details[col])
        except ValueError:
            per_det_auroc[col] = 0.5
    flagged = details[details["is_anomaly"] == 1].sort_values(  # type: ignore[reportCallIssue]
        "ensemble_score", ascending=False
    )

    return {
        "auroc": auroc,
        "n_true": n_true,
        "n_detected": int(anomalies.sum()),
        "scores": np.asarray(scores, dtype=float),
        "anomalies": np.asarray(anomalies, dtype=int),
        "threshold": ensemble.threshold,
        "per_det_auroc": per_det_auroc,
        "flagged": flagged,
        "score_cols": score_cols,
    }


def _render_2d_ensemble() -> None:
    """Ensemble results for 2D playground using selected detectors."""
    st.markdown("#### 4 · Ensemble (2D)")

    selected_names = st.session_state.get("det_names_2d", DETECTOR_DEFAULTS_LIST)
    detector_names = [n for n in DETECTOR_REGISTRY if n in selected_names and DETECTOR_REGISTRY[n].category != "Sequential"]
    if not detector_names:
        st.warning("Select at least one detector in the Detect step.")
        return

    # Ensemble mode selector (consistent with CASAS)
    card_specs = [
        {
            "id": "weighted",
            "icon": ":material/balance:",
            "title": "Weighted (soft)",
            "description": "Blends each detector's continuous score into a weighted average — uses all the information.",
            "badge": "continuous",
            "color": PRIMARY,
        },
        {
            "id": "majority",
            "icon": ":material/how_to_vote:",
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

    res = _ensemble_2d_results(
        tuple(detector_names),
        detector_params,
        dataset_key_internal,
        n_samples,
        contamination,
        ensemble_mode,
    )
    auroc = res["auroc"]
    n_true_anomalies = res["n_true"]
    n_detected = res["n_detected"]
    scores = res["scores"]
    anomalies = res["anomalies"]
    threshold = res["threshold"]
    score_cols = res["score_cols"]

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
    if threshold is not None:
        fig_timeline.add_hline(y=threshold, line={"color": "#ef4444", "width": 1.5, "dash": "dash"}, annotation_text="threshold (p90)")
    fig_timeline.update_layout(title="Ensemble Scores", xaxis_title="Sample Index", yaxis_title="Score", yaxis_range=[0, 1], height=350, showlegend=False)
    st.plotly_chart(fig_timeline, width="stretch")

    # Score distribution
    fig_hist = go.Figure(go.Histogram(x=scores, nbinsx=30, marker_color="#6366f1", name="Score Distribution"))
    if threshold is not None:
        fig_hist.add_vline(x=threshold, line={"color": "#ef4444", "width": 1.5, "dash": "dash"}, annotation_text="threshold")
    fig_hist.update_layout(title="Score Distribution", xaxis_title="Score", yaxis_title="Count", height=250)
    st.plotly_chart(fig_hist, width="stretch")

    # Per-detector AUROC (computed in the cached call)
    if score_cols:
        st.markdown("**Per-Detector AUROC**")
        auroc_data = [
            {"Detector": col.replace("_score", ""), "AUROC": f"{res['per_det_auroc'][col]:.3f}"}
            for col in score_cols
        ]
        st.dataframe(pd.DataFrame(auroc_data), hide_index=True, width="stretch")

    # Flagged points
    flagged = res["flagged"]
    st.markdown("**Flagged as Anomalous**")
    st.dataframe(flagged[["ensemble_score", "is_anomaly", *score_cols]], width="stretch")


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
        width="stretch",
        disabled=(step == 0),
        on_click=back_step,
        args=("workflow_step",),
    )
with col_right:
    if step < len(STEPS) - 1:
        st.button(
            "Continue →",
            key="next_btn",
            width="stretch",
            type="primary",
            on_click=advance_step,
            args=("workflow_step", len(STEPS)),
        )
    else:
        st.button(
            "Back to start",
            key="home_btn",
            width="stretch",
            on_click=lambda: st.session_state.update(workflow_step_current=0, workflow_step_completed=0),
        )

logger.info("Streamlit app executed correctly")
