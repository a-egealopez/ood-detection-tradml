"""CASAS track: run the ensemble pipeline per house and inspect the results.

The view owns all Streamlit widgets (sidebar configuration + result tabs) while the
actual ML work lives in ``src.pipeline``.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import metric_row, section_title  # noqa: E402
from config import db_path  # noqa: E402
from data_access import list_houses, load_house_events  # noqa: E402
from evaluation.synthetic_injection import describe_scores  # noqa: E402
from streamlit_config import (  # noqa: E402
    DETECTOR_DEFAULTS_LIST,
    DETECTOR_NAMES,
    DETECTOR_REGISTRY,
    ENSEMBLE_DEFAULTS,
    SYNTHETIC_EVALUATION_DEFAULTS,
    TRAINING_DEFAULTS,
)
from theme import ANOMALY, PRIMARY, PRIMARY_SOFT, apply_layout, display_chart  # noqa: E402


# ============================================================================
# SIDEBAR
# ============================================================================
def _render_sidebar(source: str) -> dict:
    st.sidebar.header(f"Configuration [{source}]")
    st.sidebar.caption(
        f"DB: `{db_path(source).name}`. "
        f"{'Synthetic test data.' if source == 'synthetic' else 'Real datasets.'}"
    )

    try:
        available_houses = list_houses(source)
    except Exception as e:
        st.error(f"Failed to read database ({source}): {e}")
        st.stop()

    if not available_houses:
        cmd = f"python src/ingestion/casas_loader.py --source {source}"
        st.warning(f"No data loaded. Run:\n\n```\n{cmd}\n```")
        st.stop()

    houses_to_run = st.sidebar.multiselect(
        "Houses to analyze",
        available_houses,
        default=available_houses,
    )

    st.sidebar.subheader("Detectors")
    detector_types = st.sidebar.multiselect(
        "Detector types",
        DETECTOR_NAMES,
        default=DETECTOR_DEFAULTS_LIST,
    )
    for detector in detector_types:
        st.sidebar.caption(f"*{detector}:* {DETECTOR_REGISTRY[detector].description}")

    st.sidebar.markdown("---")

    detector_params = {}
    for detector in detector_types:
        spec = DETECTOR_REGISTRY[detector]
        if not spec.params:
            continue
        st.sidebar.subheader(f"{detector}")
        for param in spec.params:
            value = st.sidebar.slider(
                param.label,
                min_value=param.min,
                max_value=param.max,
                value=param.default,
                step=param.step,
                key=f"{source}_{detector}_{param.kwarg}",
            )
            detector_params[detector] = {**detector_params.get(detector, {}), param.kwarg: value}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ensemble")

    ensemble_mode_label = st.sidebar.radio(
        "Strategy",
        ["Weighted Voting (Soft)", "Majority Voting (Hard)"],
        horizontal=True,
        key=f"ensemble_mode_{source}",
    )
    ensemble_mode = "soft" if "Weighted" in ensemble_mode_label else "hard"

    if ensemble_mode == "soft":
        weighting_scheme = st.sidebar.selectbox(
            "Score Weighting",
            ["Uniform", "Entropy-based"],
            help="Uniform = equal weight | Entropy = confident detectors weighted higher",
            key=f"weighting_scheme_{source}",
        )
    else:
        st.sidebar.caption("Majority rule (binary votes)")
        weighting_scheme = "Uniform"

    threshold_percentile = st.sidebar.slider(
        "Decision threshold (percentile)",
        ENSEMBLE_DEFAULTS["threshold_percentile"] - 40,
        ENSEMBLE_DEFAULTS["threshold_percentile"] + 9,
        ENSEMBLE_DEFAULTS["threshold_percentile"],
        step=1,
        help="Ensemble score above this percentile = anomaly",
        key=f"ensemble_threshold_{source}",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Training")

    train_split = st.sidebar.slider(
        "% data for training (per house)",
        TRAINING_DEFAULTS["train_split"] - 0.4,
        TRAINING_DEFAULTS["train_split"] + 0.2,
        TRAINING_DEFAULTS["train_split"],
        step=0.05,
        key=f"train_split_{source}",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Synthetic evaluation (no real labels)")

    contamination = st.sidebar.slider(
        "Synthetic anomalies injected (%)",
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"] - 0.10,
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"] + 0.15,
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"],
        step=0.05,
        key=f"syn_contamination_{source}",
    )

    magnitude = st.sidebar.slider(
        "Anomaly magnitude (std deviations)",
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"] - 5.0,
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"] + 4.0,
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"],
        step=0.5,
        key=f"syn_magnitude_{source}",
    )

    run_button = st.sidebar.button("Run on all selected houses", key=f"run_{source}")

    pipeline_kwargs = {
        "detector_names": detector_types,
        "detector_params": detector_params,
        "train_split": train_split,
        "weighting_scheme": weighting_scheme,
        "ensemble_mode": ensemble_mode,
        "threshold_percentile": threshold_percentile,
        "contamination": contamination,
        "magnitude": magnitude,
    }
    return {"houses": houses_to_run, "run": run_button, "pipeline_kwargs": pipeline_kwargs}


def _run_pipeline(source: str, config: dict) -> dict:
    from pipeline import run_house

    results = st.session_state.get(f"results_{source}", {})
    if not config["run"]:
        return results

    if not config["houses"]:
        st.sidebar.error("Select at least one house.")
        return results

    new_results = {}
    with st.spinner(f"Processing {len(config['houses'])} house(s)..."):
        for house_id in config["houses"]:
            df = load_house_events(source, house_id)
            result, error = run_house(house_id, df, **config["pipeline_kwargs"])
            if error:
                st.sidebar.warning(f"[{house_id}] {error}")
            else:
                new_results[house_id] = result

    st.session_state[f"results_{source}"] = new_results
    return new_results


# ============================================================================
# RESULT TABS
# ============================================================================
def _render_data_tab(results: dict, source: str) -> None:
    if not results:
        st.info("Select houses and click 'Run' in the sidebar.")
        return

    house_view = st.selectbox("House to view", list(results.keys()), key=f"data_house_{source}")
    df = results[house_view].df

    section_title(f"Raw Data - {house_view}")
    st.dataframe(df.head(20), use_container_width=True)

    date_range = (pd.to_datetime(df["timestamp"]).max() - pd.to_datetime(df["timestamp"]).min()).days
    metric_row(
        [
            ("Total events", f"{len(df):,}"),
            ("Unique sensors", f"{df['sensor_id'].nunique()}"),
            ("Date range (days)", str(date_range)),
        ]
    )

    daily_counts = df.groupby(pd.to_datetime(df["timestamp"]).dt.date).size()
    fig = go.Figure(
        go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode="lines",
            name="Events/Day",
            line={"color": PRIMARY},
        )
    )
    apply_layout(fig, "Events per Day")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="N events")
    display_chart(fig, key=f"events_day_{source}_{house_view}")


def _render_anomalies_tab(results: dict, source: str) -> None:
    if not results:
        st.info("Select houses and click 'Run' in the sidebar.")
        return

    house_view = st.selectbox("House to view", list(results.keys()), key=f"anom_house_{source}")
    r = results[house_view]
    details, anomalies, scores = r.details, r.anomalies, r.scores

    section_title(f"Anomaly Detection - {house_view}")
    metric_row(
        [
            ("Total Anomalies", str(int(anomalies.sum()))),
            ("Anomaly Rate", f"{anomalies.mean() * 100:.2f}%"),
            ("Avg Score", f"{scores.mean():.3f}"),
        ]
    )

    fig_timeline = go.Figure(
        go.Scatter(
            x=details["date"],
            y=details["ensemble_score"],
            mode="markers",
            marker={
                "color": anomalies,
                "colorscale": [[0, PRIMARY_SOFT], [1, ANOMALY]],
                "size": 8,
                "showscale": False,
            },
            name="Ensemble Score",
        )
    )
    apply_layout(fig_timeline, "Timeline: ensemble scores (red = anomaly)", height=460)
    display_chart(fig_timeline, key=f"timeline_{source}_{house_view}")

    fig_hist = go.Figure(
        go.Histogram(x=scores, nbinsx=30, marker_color=PRIMARY_SOFT, name="Score Distribution")
    )
    apply_layout(fig_hist, "Score Distribution", height=320)
    display_chart(fig_hist, key=f"hist_{source}_{house_view}")

    section_title("Days flagged as anomalous")
    anomalous_days = details[details["is_anomaly"] == 1].sort_values(
        "ensemble_score", ascending=False
    )
    st.dataframe(anomalous_days, use_container_width=True)

    score_cols = [
        c for c in details.columns if c.endswith("_score") and c != "ensemble_score"
    ]
    if score_cols:
        fig_corr = go.Figure(
            go.Heatmap(
                z=details[score_cols].corr().values,
                x=score_cols,
                y=score_cols,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
        )
        apply_layout(fig_corr, "Detector Agreement (higher = more aligned)", height=400)
        display_chart(fig_corr, key=f"corr_{source}_{house_view}")

    csv = details.to_csv(index=False)
    st.download_button(
        "Download results (CSV)",
        data=csv,
        file_name=f"anomaly_predictions_{house_view}.csv",
        mime="text/csv",
    )


def _render_metrics_tab(results: dict, source: str) -> None:
    section_title("How well does each house detect? (no real labels)")
    st.caption(
        "Synthetic anomalies are injected and we measure if the ensemble detects them."
    )

    if not results:
        st.info("Select houses and click 'Run' in the sidebar.")
        return

    rows = []
    for house_id, r in results.items():
        row = {"house_id": house_id, "n_holdout": r.n_holdout}
        row.update(describe_scores(r.scores, r.anomalies))

        sm = r.synthetic_metrics
        if sm is not None:
            row.update(
                {
                    "synthetic_precision": sm["precision"],
                    "synthetic_recall": sm["recall"],
                    "synthetic_f1": sm["f1"],
                    "synthetic_auroc": sm["auroc"],
                }
            )
        else:
            row.update(
                {
                    "synthetic_precision": None,
                    "synthetic_recall": None,
                    "synthetic_f1": None,
                    "synthetic_auroc": None,
                }
            )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, use_container_width=True)

    valid = comparison.dropna(subset=["synthetic_auroc"])
    if not valid.empty:
        fig_auroc = go.Figure(
            go.Bar(x=valid["house_id"], y=valid["synthetic_auroc"], marker_color=PRIMARY)
        )
        apply_layout(fig_auroc, "AUROC vs Synthetic Anomalies", height=400)
        fig_auroc.update_yaxes(title_text="AUROC", range=[0, 1.05])
        display_chart(fig_auroc, key=f"auroc_{source}")

    too_small = [h for h in results if results[h].synthetic_metrics is None]
    if too_small:
        st.warning(f"No synthetic metrics for {', '.join(too_small)}: holdout too small.")


def _render_info_tab() -> None:
    st.markdown(
        """
        ### Ensemble Strategy

        Each detector emits a continuous anomaly score in [0, 1]. The ensemble combines
        them with a **weighted sum rule**:

        $$S_{ensemble}(x) = \\sum_{i=1}^{n} w_i \\cdot s_i(x)$$

        **Weighting schemes**:
        - **Uniform**: all detectors are equally important.
        - **Entropy-based**: detectors with lower uncertainty get higher weight
          (confidence = 1 / (1 + entropy(scores))).

        The anomaly threshold is a percentile of the ensemble score computed on the
        training set.

        ---

        ### Why an Ensemble?

        | Detector | Strength | Weakness |
        |----------|----------|----------|
        | **Isolation Forest** | Scalable, no distribution assumption | Struggles with correlated features |
        | **Mahalanobis** | Respects covariance | Assumes Gaussian, sensitive to outliers in train |
        | **LOF** | Detects local density anomalies | Slow (O(n²)), parameter-sensitive |
        | **One-Class SVM** | Non-convex boundary, kernels | Expensive training |
        | **HMM** | Temporal patterns | Assumes Gaussian per state, expensive |

        Combining perspectives makes the detection robust to model misspecification.

        ---

        ### Feature Engineering

        Daily features extracted (window aggregation):
        **n_events** - **n_sensors** - **activity_hours** - **avg_event_gap_minutes** -
        **peak_hour** - **night_activity** - **event_frequency_std** -
        **entropy_hourly** - **entropy_sensor**.

        **Normalization**: z-score, fitted on training data only.

        ---

        ### Evaluation Without Real Labels

        Since the CASAS datasets have no anomaly annotations:

        1. Train on ~70% of the daily features.
        2. Keep ~30% as holdout.
        3. Inject synthetic anomalies (±`magnitude` std on a subset of features, on ~
           `contamination` of the holdout).
        4. Compute Precision, Recall, F1 and AUROC against the synthetic labels.

        **AUROC interpretation**: above 0.75 good discrimination, 0.7-0.75 acceptable,
        below 0.7 the ensemble is failing on synthetic anomalies - tune parameters.

        **Limitation**: real anomalies may differ from synthetic ones; this is proxy
        evaluation for hyperparameter tuning.
        """
    )


# ============================================================================
# VIEW ENTRY POINT
# ============================================================================
def render_casas_view(source: str) -> None:
    section_title(f"Feature Extraction & Anomaly Detection - {source.title()} Data")
    st.caption(
        "Each house is trained independently; synthetic anomaly injection provides the "
        "evaluation metrics since no real labels exist."
    )

    config = _render_sidebar(source)
    results = _run_pipeline(source, config)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data", "Anomalies by House", "Metrics (no labels)", "Info"]
    )
    with tab1:
        _render_data_tab(results, source)
    with tab2:
        _render_anomalies_tab(results, source)
    with tab3:
        _render_metrics_tab(results, source)
    with tab4:
        _render_info_tab()
