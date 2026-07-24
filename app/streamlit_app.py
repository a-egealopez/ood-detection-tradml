import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from ingestion.sqlite_manager import SQLiteDataManager
from features import FeatureScaler, TemporalFeatureExtractor
from detectors import (
    IsolationForestDetector,
    ExtendedIForestDetector,
    MahalanobisDetector,
    EllipticEnvelopeDetector,
    RobustCovarianceDetector,
    KNNDetector,
    OCSVMDetector,
    LOFDetector,
    HMMDetector,
    HawkesDetector,
)
from detectors.ensemble import EnsembleDetector
from evaluation.synthetic_injection import evaluate_with_synthetic_anomalies, describe_scores
from config import setup_logging, DB_PATH
from streamlit_config import (
    DETECTOR_PARAMS,
    ENSEMBLE_DEFAULTS,
    DETECTOR_DEFAULTS_LIST,
    TRAINING_DEFAULTS,
    SYNTHETIC_EVALUATION_DEFAULTS,
    DETECTOR_DISPLAY_ORDER,
    DETECTOR_DESCRIPTIONS,
    DETECTOR_CATEGORIES,
    DETECTOR_PARAM_MAP,
)
from teaching_tab import render_teaching_tab
from feature_extraction_tab import render_feature_extraction_tab

st.set_page_config(page_title="Anomaly Detection - Health IoT", layout="wide")
logger = setup_logging()

st.title("🔍 Anomaly Detection: Health IoT (CASAS)")
st.markdown(
    "**Unsupervised ensemble**: vectorial + sequential detectors. "
    "Each house trained independently using synthetic anomaly injection for evaluation."
)

# ============================================================================
# SELECTOR GLOBAL ANTES DE TABS: Synthetic vs Real
# ============================================================================
col_global_1, col_global_2 = st.columns([2, 1])

with col_global_1:
    data_mode = st.radio(
        "Choose data source:",
        ["🎓 Teaching: Synthetic Datasets", "🏠 Synthetic CASAS Data", "🏠 Real CASAS Data", "📚 Documentation"],
        horizontal=True,
        index=1,
    )

# ============================================================================
# TAB: TEACHING WITH SYNTHETIC DATASETS (Datasets Didácticos)
# ============================================================================
if data_mode == "🎓 Teaching: Synthetic Datasets":
    render_teaching_tab()

# ============================================================================
# TAB: FEATURE EXTRACTION (Nueva pestaña de extracción de características)
# ============================================================================
elif data_mode == "🏠 Synthetic CASAS Data":
    st.markdown("---")
    st.subheader("Feature Extraction & Anomaly Detection on Synthetic Data")
    
    # Selector de pestaña dentro de Synthetic
    feat_mode = st.radio(
        "View mode (Synthetic):",
        ["Feature Extraction Tutorial", "Anomaly Detection Pipeline"],
        horizontal=True,
        index=1,
    )
    
    if feat_mode == "Feature Extraction Tutorial":
        render_feature_extraction_tab()
    else:
        # Pipeline de anomalías sintético
        _run_anomaly_pipeline("synthetic")

# ============================================================================
# TAB: REAL CASAS DATA (Datos Reales de IoT)
# ============================================================================
elif data_mode == "🏠 Real CASAS Data":
    st.markdown("---")
    st.subheader("Feature Extraction & Anomaly Detection on Real Data")
    
    # Selector de pestaña dentro de Real
    feat_mode_real = st.radio(
        "View mode (Real):",
        ["Feature Extraction Tutorial", "Anomaly Detection Pipeline"],
        horizontal=True,
        index=1,
    )
    
    if feat_mode_real == "Feature Extraction Tutorial":
        render_feature_extraction_tab()
    else:
        # Pipeline de anomalías real
        run_anomaly_pipeline("real")

# ============================================================================
# TAB: DOCUMENTATION
# ============================================================================
elif data_mode == "📚 Documentation":
    st.subheader("📚 Documentation & Architecture")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application is an **unsupervised anomaly detection system** for IoT sensor data
    (CASAS - Center for Advanced Studies in Adaptive Systems). It combines multiple
    detection algorithms in an ensemble for robustness.
    
    ### 📂 Project Structure
    
    ```
    ood-detection-tradml/
    ├── app/
    │   ├── streamlit_app.py          # Main app
    │   ├── streamlit_config.py       # Configuration & defaults
    │   ├── teaching_tab.py           # Teaching tab (synthetic datasets)
    │   └── feature_extraction_tab.py # Feature extraction tutorial
    ├── src/
    │   ├── config.py                 # Path & logging config
    │   ├── detectors/                # Anomaly detection algorithms
    │   ├── features/                 # Feature extraction
    │   │   ├── scaler.py
    │   │   ├── temporal_features.py
    │   │   └── event_driven_extractors.py  # NEW: 3 methods for event-driven TS
    │   ├── ingestion/                # Data loading
    │   ├── evaluation/               # Metrics & synthetic injection
    │   └── teaching/                 # Teaching datasets & viz
    ├── data/
    │   ├── real/                     # CASAS datasets (4 houses)
    │   └── synthetic/                # Generated test data
    └── scripts/                      # Utilities & evaluation
    ```
    
    ### 🔍 Detector Types
    
    **Vectorial (Distance/Density-based)**:
    - Isolation Forest: Random partitioning
    - Extended IForest: Sliced paths (high-dimensional)
    - Mahalanobis: Distance metric respecting covariance
    - Elliptic Envelope: Robust Gaussian fitting
    - Robust Covariance: Minimum Covariance Determinant
    - KNN: k-nearest neighbor distance
    - One-Class SVM: Non-convex boundary learning
    - LOF: Local density factor
    
    **Sequential (Time-series)**:
    - HMM: Hidden Markov Model transitions
    - Hawkes: Self-exciting point process
    
    ### 🎓 Teaching Tab
    
    Interactive exploration of anomaly detectors on synthetic datasets from scikit-learn:
    - Compare up to 8 algorithms simultaneously
    - Adjust parameters in real-time
    - Visualize detector behavior on different geometries
    - Educational insights on algorithm strengths/weaknesses
    
    ### 🧬 Feature Extraction
    
    Three methods from event-driven time series literature:
    
    1. **Window Aggregation** → Contextual + Colectiva (day-level)
       - Estadísticas por ventana: conteos, entropía, % nocturno
       - Detecta días atípicos como conjunto
    
    2. **Inter-Event Interval (IEI)** → Puntual (crudo) / Colectiva (agregado)
       - Analiza tiempo entre eventos como point process
       - CV y Fano factor para regualridad/burstiness
    
    3. **N-gram Transition (Markov)** → Colectiva de secuencia
       - Matriz de transición (bigramas) de sensores
       - Detecta rutinas que rompen el patrón habitual
    
    ### 📊 Key Metrics
    
    - **Precision**: TP / (TP + FP)
    - **Recall**: TP / (TP + FN)
    - **F1-Score**: Harmonic mean of precision and recall
    - **AUROC**: Area under ROC curve (0.5=random, 1.0=perfect)
    
    ### 🔗 Ensemble Strategies
    
    **Soft Voting (Weighted Sum Rule)**:
    $$S_{ensemble}(x) = \\sum_{i=1}^{n} w_i \\cdot s_i(x)$$
    
    - Uniform: All detectors equally important
    - Entropy-based: Detectors with lower uncertainty get higher weight
    
    **Hard Voting (Majority Rule)**:
    - Each detector emits a binary vote
    - Robust to outliers in individual scores
    - Loses information from continuous scores
    
    ### ⚠️ Note on Evaluation
    
    Since CASAS datasets lack real anomaly labels, we use **synthetic anomaly injection**:
    1. Train on 70% of data
    2. Split 30% as holdout
    3. Inject synthetic anomalies (±6σ perturbations on ~15% of holdout)
    4. Compute Precision, Recall, F1, AUROC vs. synthetic labels
    
    **Limitation**: Real anomalies may differ from synthetic patterns.
    This is a proxy evaluation to tune hyperparameters.
    """)


# ============================================================================
# HELPER FUNCTION: Pipeline de Anomalías (reutilizable)
# ============================================================================

def _run_anomaly_pipeline(source: str):
    """
    Pipeline completo de anomalía detection para synthetic o real.
    
    source: "synthetic" o "real"
    
    Estructura: todo el sidebar va aquí, y las tabs de presentación abajo.
    """

    def db_path_for_source(s: str) -> Path:
        if s == "synthetic":
            return DB_PATH.parent / "synthetic" / "sensor_data.db"
        else:
            return DB_PATH

    @st.cache_data
    def load_available_houses(s: str):
        db = SQLiteDataManager(str(db_path_for_source(s)))
        db.connect()
        houses = db.list_houses()
        db.close()
        return houses

    @st.cache_data
    def load_house_data(house_id: str, s: str):
        db = SQLiteDataManager(str(db_path_for_source(s)))
        db.connect()
        df = db.query_house(house_id)
        db.close()
        return df

    # ========================================================================
    # SIDEBAR: ÚNICO BLOQUE DE CONFIGURACIÓN
    # ========================================================================
    st.sidebar.header(f"⚙️ Configuration [{source.upper()}]")
    st.sidebar.caption(
        f"DB: `{db_path_for_source(source).name}`. "
        f"{'Synthetic test data.' if source == 'synthetic' else 'Real datasets.'}"
    )

    try:
        available_houses = load_available_houses(source)
    except Exception as e:
        st.error(f"Failed to read database ({source}): {e}")
        st.stop()

    if not available_houses:
        cmd = f"python src/ingestion/casas_loader.py --source {source}"
        st.warning(f"No data loaded. Run:\n\n```\n{cmd}\n```")
        st.stop()

    houses_to_run = st.sidebar.multiselect(
        "🏠 Houses to analyze",
        available_houses,
        default=available_houses,
    )

    st.sidebar.subheader("🔧 Select Detectors")
    detector_types = st.sidebar.multiselect(
        "Detector types",
        DETECTOR_DISPLAY_ORDER,
        default=DETECTOR_DEFAULTS_LIST,
    )

    for detector in detector_types:
        st.sidebar.caption(f"*{detector}:* {DETECTOR_DESCRIPTIONS[detector]}")

    st.sidebar.markdown("---")

    detector_params = {}
    for detector_name in detector_types:
        if detector_name in DETECTOR_PARAMS:
            st.sidebar.subheader(f"⚙️ {detector_name}")
            for param_name, param_config in DETECTOR_PARAMS[detector_name].items():
                value = st.sidebar.slider(
                    param_name,
                    min_value=param_config["min"],
                    max_value=param_config["max"],
                    value=param_config["default"],
                    step=param_config["step"],
                    key=f"{source}_{detector_name}_{param_name}",
                )
                detector_params[f"{detector_name}_{param_name}"] = value

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Ensemble Config")

    ensemble_mode = st.sidebar.radio(
        "Ensemble Strategy",
        ["Weighted Voting (Soft)", "Majority Voting (Hard)"],
        horizontal=True,
        key=f"ensemble_mode_{source}",
    )
    ensemble_mode = "soft" if "Weighted" in ensemble_mode else "hard"

    if ensemble_mode == "soft":
        st.sidebar.caption("Soft Voting: weighted sum of scores (0-1)")
        weighting_scheme = st.sidebar.selectbox(
            "Score Weighting",
            ["Uniform", "Entropy-based", "Performance-based (mock)"],
            help="Uniform = equal weight | Entropy = confident detectors weighted higher",
            key=f"weighting_scheme_{source}",
        )
    else:
        st.sidebar.caption("Hard Voting: Majority rule (binary)")
        weighting_scheme = "Uniform"

    ensemble_decision_threshold = st.sidebar.slider(
        "Decision Threshold (percentile)",
        ENSEMBLE_DEFAULTS["threshold_percentile"] - 40,
        ENSEMBLE_DEFAULTS["threshold_percentile"] + 9,
        ENSEMBLE_DEFAULTS["threshold_percentile"],
        step=1,
        help="Ensemble score > this percentile → anomaly",
        key=f"ensemble_threshold_{source}",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Training")

    train_split = st.sidebar.slider(
        "% data for training (per house)",
        TRAINING_DEFAULTS["train_split"] - 0.4,
        TRAINING_DEFAULTS["train_split"] + 0.2,
        TRAINING_DEFAULTS["train_split"],
        step=0.05,
        key=f"train_split_{source}",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Synthetic Evaluation (no real labels)")

    syn_contamination = st.sidebar.slider(
        "Synthetic anomalies injected (%)",
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"] - 0.10,
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"] + 0.15,
        SYNTHETIC_EVALUATION_DEFAULTS["contamination"],
        step=0.05,
        key=f"syn_contamination_{source}",
    )

    syn_magnitude = st.sidebar.slider(
        "Anomaly magnitude (std deviations)",
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"] - 5.0,
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"] + 4.0,
        SYNTHETIC_EVALUATION_DEFAULTS["magnitude"],
        step=0.5,
        key=f"syn_magnitude_{source}",
    )

    run_button = st.sidebar.button(f"🚀 Run on all selected houses", key=f"run_{source}")

    # ========================================================================
    # FUNCIONES AUXILIARES
    # ========================================================================

    def build_detectors(detector_types, detector_params):
        detectors = []
        detector_map = {
            "Isolation Forest": IsolationForestDetector,
            "Extended IForest": ExtendedIForestDetector,
            "Mahalanobis": MahalanobisDetector,
            "Elliptic Envelope": EllipticEnvelopeDetector,
            "Robust Covariance": RobustCovarianceDetector,
            "KNN": KNNDetector,
            "OC-SVM": OCSVMDetector,
            "LOF": LOFDetector,
            "HMM": HMMDetector,
            "Hawkes": HawkesDetector,
        }

        for detector_name in detector_types:
            if detector_name not in detector_map:
                continue
            detector_class = detector_map[detector_name]
            kwargs = {}
            if detector_name in DETECTOR_PARAM_MAP:
                for param_key, param_attr in DETECTOR_PARAM_MAP[detector_name].items():
                    full_key = f"{detector_name}_{param_key}"
                    if full_key in detector_params:
                        kwargs[param_attr] = detector_params[full_key]
            detectors.append(detector_class(**kwargs))
        return detectors

    def run_pipeline_for_house(house_id: str):
        df_house = load_house_data(house_id, source)
        if df_house.empty:
            return None, f"No data for '{house_id}'."

        extractor = TemporalFeatureExtractor()
        X, dates = extractor.extract(df_house)

        if len(X) < 10:
            return None, f"Too few days in '{house_id}' ({len(X)})."

        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)

        split_idx = max(1, int(len(X_scaled) * train_split))
        X_train, X_holdout = X_scaled[:split_idx], X_scaled[split_idx:]

        detectors = build_detectors(detector_types, detector_params)
        if not detectors:
            return None, "Select at least one detector"

        n_detectors = len(detectors)

        if weighting_scheme == "Uniform":
            weights = np.array([1.0 / n_detectors] * n_detectors)
        elif weighting_scheme == "Entropy-based":
            weights = []
            for detector in detectors:
                detector.fit(X_train)
                _, scores_train = detector.predict(X_train)
                entropy = -np.mean(
                    scores_train * np.log(scores_train + 1e-10) +
                    (1 - scores_train) * np.log(1 - scores_train + 1e-10)
                )
                confidence = 1.0 / (1.0 + entropy)
                weights.append(confidence)
            weights = np.array(weights)
            weights = weights / weights.sum()
        else:
            weights = np.random.dirichlet(np.ones(n_detectors))

        ensemble = EnsembleDetector(
            detectors=detectors,
            weights=weights,
            ensemble_mode=ensemble_mode,
            ensemble_threshold_percentile=ensemble_decision_threshold,
        )
        ensemble.fit(X_train)
        anomalies, scores, details = ensemble.predict(X_scaled)
        details["date"] = dates

        synthetic_metrics = None
        if len(X_holdout) >= 5:
            synthetic_metrics = evaluate_with_synthetic_anomalies(
                ensemble, X_holdout, contamination=syn_contamination, magnitude=syn_magnitude
            )

        return {
            "df": df_house,
            "details": details,
            "anomalies": anomalies,
            "scores": scores,
            "ensemble": ensemble,
            "scaler": scaler,
            "n_holdout": len(X_holdout),
            "synthetic_metrics": synthetic_metrics,
        }, None

    # ========================================================================
    # EJECUCIÓN DEL PIPELINE
    # ========================================================================

    if run_button:
        if not houses_to_run:
            st.sidebar.error("Select at least one house.")
        else:
            results = {}
            with st.spinner(f"Processing {len(houses_to_run)} house(es)..."):
                for house_id in houses_to_run:
                    result, error = run_pipeline_for_house(house_id)
                    if error:
                        st.sidebar.warning(f"[{house_id}] {error}")
                    else:
                        results[house_id] = result
            st.session_state[f"results_{source}"] = results

    results = st.session_state.get(f"results_{source}", {})
    processed_houses = list(results.keys())

    # ========================================================================
    # TABS DE PRESENTACIÓN
    # ========================================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data", "🚨 Anomalies by House", "📐 Metrics (no labels)", "📋 Info"]
    )

    with tab1:
        if not processed_houses:
            st.info("Select houses and click 'Run' in the sidebar.")
        else:
            house_view = st.selectbox("House to view", processed_houses, key=f"data_house_{source}")
            df = results[house_view]["df"]

            st.subheader(f"Raw Data — {house_view}")
            st.dataframe(df.head(20), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total events", f"{len(df):,}")
            col2.metric("Unique sensors", df["sensor_id"].nunique())
            col3.metric("Date range (days)", f"{(df['timestamp'].max() - df['timestamp'].min()).days}")

            daily_counts = df.groupby(df["timestamp"].dt.date).size()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts.values, mode="lines", name="Events/Day"))
            fig.update_layout(title="Events per Day", height=400, xaxis_title="Date", yaxis_title="N events")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not processed_houses:
            st.info("Select houses and click 'Run' in the sidebar.")
        else:
            house_view = st.selectbox("House to view", processed_houses, key=f"anom_house_{source}")
            r = results[house_view]
            details, anomalies, scores = r["details"], r["anomalies"], r["scores"]

            st.subheader(f"🚨 Anomaly Detection — {house_view}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Anomalies", int(anomalies.sum()))
            col2.metric("Anomaly Rate", f"{anomalies.mean() * 100:.2f}%")
            col3.metric("Avg Score", f"{scores.mean():.3f}")

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=details["date"], y=details["ensemble_score"], mode="markers",
                marker=dict(
                    color=anomalies, colorscale=[[0, "steelblue"], [1, "crimson"]], size=8, showscale=False
                ),
                name="Ensemble Score",
            ))
            fig_timeline.update_layout(title="Timeline: Ensemble Scores (red = anomaly)", height=450)
            st.plotly_chart(fig_timeline, use_container_width=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=scores, nbinsx=30, name="Score Distribution"))
            fig_hist.update_layout(title="Score Distribution", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Days flagged as anomalous")
            anomalous_days = details[details["is_anomaly"] == 1].sort_values("ensemble_score", ascending=False)
            st.dataframe(anomalous_days, use_container_width=True)

            score_cols = [c for c in details.columns if c.endswith("_score") and c != "ensemble_score"]
            if score_cols:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=details[score_cols].corr().values, x=score_cols, y=score_cols,
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                ))
                fig_corr.update_layout(
                    title="Detector Agreement (higher = more aligned)",
                    height=400,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            csv = details.to_csv(index=False)
            st.download_button(
                "📥 Download results (CSV)", data=csv,
                file_name=f"anomaly_predictions_{house_view}.csv", mime="text/csv",
            )

    with tab3:
        st.subheader("📐 How well does each house detect? (no real labels)")
        st.caption(
            "Synthetic anomalies are injected and we measure if the ensemble detects them."
        )

        if not processed_houses:
            st.info("Select houses and click 'Run' in the sidebar.")
        else:
            rows = []
            for house_id in processed_houses:
                r = results[house_id]
                row = {"house_id": house_id, "n_holdout": r["n_holdout"]}
                row.update(describe_scores(r["scores"], r["anomalies"]))

                sm = r["synthetic_metrics"]
                if sm is not None:
                    row.update({
                        "synthetic_precision": sm["precision"],
                        "synthetic_recall": sm["recall"],
                        "synthetic_f1": sm["f1"],
                        "synthetic_auroc": sm["auroc"],
                    })
                else:
                    row.update({
                        "synthetic_precision": None, "synthetic_recall": None,
                        "synthetic_f1": None, "synthetic_auroc": None,
                    })
                rows.append(row)

            comparison = pd.DataFrame(rows)
            st.dataframe(comparison, use_container_width=True)

            valid = comparison.dropna(subset=["synthetic_auroc"])
            if not valid.empty:
                fig_auroc = go.Figure(data=[go.Bar(x=valid["house_id"], y=valid["synthetic_auroc"])])
                fig_auroc.update_layout(
                    title="AUROC vs Synthetic Anomalies",
                    yaxis_title="AUROC", height=400, yaxis_range=[0, 1.05],
                )
                st.plotly_chart(fig_auroc, use_container_width=True)

            too_small = [h for h in processed_houses if results[h]["synthetic_metrics"] is None]
            if too_small:
                st.warning(
                    f"No synthetic metrics for {', '.join(too_small)}: holdout too small."
                )

    with tab4:
        st.subheader("📊 Ensemble Strategies & Mathematical Background")

        st.markdown("""
        ### Soft Voting (Weighted Sum Rule)

        Each detector produces a continuous anomaly score in [0, 1]. The ensemble aggregates:

        $$S_{ensemble}(x) = \\sum_{i=1}^{n} w_i \\cdot s_i(x)$$

        where $w_i$ are normalized weights summing to 1.

        **Weighting schemes**:
        - **Uniform**: All detectors equally important
        - **Entropy-based**: Detectors with lower uncertainty get higher weight
          - Confidence = 1 / (1 + entropy(scores))
        - **Performance-based**: Weights based on validation metrics (not implemented here)

        **Threshold**: Anomalies are flagged when ensemble score > percentile of training scores.

        ---

        ### Hard Voting (Majority Rule)

        Each detector emits a binary vote based on its individual threshold:

        $$V_ensemble(x) = \\sum_{i=1}^{n} w_i \\cdot \\mathbb{1}[s_i(x) > \\tau_i]$$

        where $\\tau_i = median(s_i(X_{train}))$.

        **Advantage**: Robust to outliers in individual detector scores.
        **Disadvantage**: Loses information from continuous scores.

        ---

        ### Why Ensemble?

        | Detector | Strength | Weakness |
        |----------|----------|----------|
        | **Isolation Forest** | Scalable, no distribution assumption | Struggles with correlated features |
        | **Mahalanobis** | Respects covariance | Assumes Gaussian, sensitive to outliers in train |
        | **LOF** | Detects local density anomalies | Slow (O(n²)), parameter-sensitive |
        | **One-Class SVM** | Non-convex boundary, kernels | Expensive training |
        | **HMM** | Temporal patterns | Assumes Gaussian per state, expensive |

        Ensemble combines perspectives → robust to model misspecification.

        ---

        ### Feature Engineering

        Daily features extracted (Window Aggregation):
        1. **n_events**: Total sensor events
        2. **n_sensors**: Unique sensors activated
        3. **activity_hours**: Hours with activity
        4. **avg_event_gap_minutes**: Average time between events
        5. **peak_hour**: Most active hour
        6. **night_activity**: Fraction of events (22:00-08:00)
        7. **event_frequency_std**: Std of events across sensors
        8. **entropy_hourly**: Shannon entropy of hourly distribution
        9. **entropy_sensor**: Shannon entropy across sensors

        **Normalization**: Z-score (fit on training data, apply to holdout/test).

        ---

        ### Evaluation Without Real Labels

        Since CASAS datasets lack real anomaly annotations:

        1. **Train** ensemble on 70% of data
        2. **Split** 30% as holdout
        3. **Inject** synthetic anomalies:
           - Select ~15% of holdout samples
           - Perturb ±6σ in random features
        4. **Predict** on perturbed holdout
        5. **Compute** Precision, Recall, F1, AUROC vs. synthetic labels

        **Limitation**: Real anomalies may differ from synthetic. This is proxy evaluation.

        **AUROC Interpretation**:
        - **> 0.75**: Good discrimination
        - **0.7-0.75**: Acceptable
        - **< 0.7**: Ensemble fails on synthetic anomalies; tune parameters
        """)


logger.info("Streamlit app executed correctly")