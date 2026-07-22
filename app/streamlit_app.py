import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ingestion.sqlite_manager import SQLiteDataManager
from features import FeatureScaler, TemporalFeatureExtractor
from detectors import ZScoreDetector, IsolationForestDetector, PCAReconstructionDetector, EnsembleDetector
from evaluation.synthetic_injection import evaluate_with_synthetic_anomalies, describe_scores
from config import setup_logging, db_path

st.set_page_config(page_title="Anomaly Detection - Health IoT", layout="wide")
logger = setup_logging()

st.title("🔍 Anomaly Detection: Health IoT (CASAS)")
st.markdown(
    "Ensemble no supervisado: **Z-Score + Isolation Forest + PCA Reconstruction Error**. "
    "Cada casa se entrena y evalúa de forma independiente, con el mismo proceso."
)

st.sidebar.header("⚙️ Configuración")

source_label = st.sidebar.radio(
    "🗂️ Fuente de datos", ["Real", "Sintética (prueba)"], horizontal=True,
)
source = "real" if source_label == "Real" else "synthetic"
st.sidebar.caption(
    f"BD activa: `{db_path(source).name}`. Real y sintética viven en bases de "
    f"datos separadas, nunca se mezclan."
)


@st.cache_data
def load_available_houses(source: str):
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    houses = db.list_houses()
    db.close()
    return houses


@st.cache_data
def load_house_data(house_id: str, source: str):
    db = SQLiteDataManager(str(db_path(source)))
    db.connect()
    df = db.query_house(house_id)
    db.close()
    return df


try:
    available_houses = load_available_houses(source)
except Exception as e:
    st.error(f"No se pudo leer la base de datos ({source}): {e}")
    st.stop()

if not available_houses:
    load_cmd = f"python src/ingestion/casas_loader.py --source {source}"
    if source == "synthetic":
        st.warning(
            f"No hay datos sintéticos cargados. Genera y carga con:\n\n"
            f"```\npython scripts/generate_test_fixtures.py\n{load_cmd}\n```"
        )
    else:
        st.warning(f"No hay datos reales cargados. Corre primero:\n\n```\n{load_cmd}\n```")
    st.stop()

houses_to_run = st.sidebar.multiselect(
    "🏠 Casas a analizar", available_houses, default=available_houses
)

st.sidebar.subheader("Detectores")
zscore_thresh = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.1)
iforest_contam = st.sidebar.slider("IForest Contamination", 0.01, 0.20, 0.05, step=0.01)
pca_comps = st.sidebar.slider("PCA Components", 2, 9, 5)
ensemble_thresh_pctl = st.sidebar.slider("Ensemble Threshold (percentile)", 50, 99, 90)
train_split = st.sidebar.slider("% datos para entrenamiento (por casa)", 0.3, 0.9, 0.7, step=0.05)

st.sidebar.subheader("Evaluación sintética (sin etiquetas reales)")
syn_contamination = st.sidebar.slider("Anomalías sintéticas inyectadas (%)", 0.05, 0.30, 0.15, step=0.05)
syn_magnitude = st.sidebar.slider("Magnitud de la anomalía sintética (desv. estándar)", 1.0, 10.0, 6.0, step=0.5)

run_button = st.sidebar.button("🚀 Ejecutar en todas las casas seleccionadas")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Nota: estos datasets no tienen etiquetas de anomalía reales, así que "
    "'Tasa de Anomalías' es una detección no supervisada, no validada contra "
    "un ground-truth. La pestaña '📐 Métricas' inyecta anomalías sintéticas "
    "de magnitud conocida en el holdout de cada casa para poder calcular "
    "precision/recall/F1/AUROC de verdad, sin necesitar etiquetas reales."
)


def run_pipeline_for_house(house_id: str):
    df_house = load_house_data(house_id, source)
    if df_house.empty:
        return None, f"Sin datos para '{house_id}'."

    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df_house)

    if len(X) < 10:
        return None, f"Muy pocos días de datos en '{house_id}' ({len(X)}) para un análisis significativo."

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = max(1, int(len(X_scaled) * train_split))
    X_train, X_holdout = X_scaled[:split_idx], X_scaled[split_idx:]

    ensemble = EnsembleDetector(
        detectors=[
            ZScoreDetector(threshold=zscore_thresh),
            IsolationForestDetector(contamination=iforest_contam),
            PCAReconstructionDetector(n_components=pca_comps),
        ],
        ensemble_threshold_percentile=ensemble_thresh_pctl,
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


if run_button:
    if not houses_to_run:
        st.sidebar.error("Selecciona al menos una casa.")
    else:
        results = {}
        with st.spinner(f"Procesando {len(houses_to_run)} casa(s), cada una de forma independiente..."):
            for house_id in houses_to_run:
                result, error = run_pipeline_for_house(house_id)
                if error:
                    st.sidebar.warning(f"[{house_id}] {error}")
                else:
                    results[house_id] = result
        st.session_state["results"] = results
        st.session_state["results_source"] = source

results = st.session_state.get("results", {}) if st.session_state.get("results_source") == source else {}
processed_houses = list(results.keys())

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Datos", "🚨 Anomalías por casa", "📐 Métricas (sin etiquetas)", "📋 Info"]
)

with tab1:
    if not processed_houses:
        st.info("Selecciona casas y haz click en 'Ejecutar' en la barra lateral.")
    else:
        house_view = st.selectbox("Casa a visualizar", processed_houses, key="data_house")
        df = results[house_view]["df"]

        st.subheader(f"Raw Data — {house_view}")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total eventos", f"{len(df):,}")
        col2.metric("Sensores únicos", df["sensor_id"].nunique())
        col3.metric("Rango de fechas", f"{(df['timestamp'].max() - df['timestamp'].min()).days} días")

        daily_counts = df.groupby(df["timestamp"].dt.date).size()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts.values, mode="lines", name="Eventos/Día"))
        fig.update_layout(title="Eventos por Día", height=400, xaxis_title="Fecha", yaxis_title="N° eventos")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not processed_houses:
        st.info(" Selecciona casas y haz click en 'Ejecutar' en la barra lateral.")
    else:
        house_view = st.selectbox("Casa a visualizar", processed_houses, key="anom_house")
        r = results[house_view]
        details, anomalies, scores = r["details"], r["anomalies"], r["scores"]

        st.subheader(f" Anomaly Detection — {house_view} (modelo propio, sin datos de otras casas)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomalías", int(anomalies.sum()))
        col2.metric("Tasa de Anomalías", f"{anomalies.mean() * 100:.2f}%")
        col3.metric("Score Promedio", f"{scores.mean():.3f}")

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=details["date"], y=details["ensemble_score"], mode="markers",
            marker=dict(
                color=anomalies, colorscale=[[0, "steelblue"], [1, "crimson"]], size=8, showscale=False
            ),
            name="Ensemble Score",
        ))
        fig_timeline.update_layout(title="Timeline: Scores del Ensemble (rojo = anomalía)", height=450)
        st.plotly_chart(fig_timeline, use_container_width=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=scores, nbinsx=30, name="Distribución de Scores"))
        fig_hist.update_layout(title="Distribución de Scores", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Días marcados como anómalos")
        anomalous_days = details[details["is_anomaly"] == 1].sort_values("ensemble_score", ascending=False)
        st.dataframe(anomalous_days, use_container_width=True)

        score_cols = [c for c in details.columns if c.endswith("_score")]
        fig_corr = go.Figure(data=go.Heatmap(
            z=details[score_cols].corr().values, x=score_cols, y=score_cols,
            colorscale="RdBu_r", zmin=-1, zmax=1,
        ))
        fig_corr.update_layout(
            title="Correlación entre los 3 detectores (más alto = más de acuerdo entre sí)",
            height=400,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        csv = details.to_csv(index=False)
        st.download_button(
            "📥 Descargar resultados (CSV)", data=csv,
            file_name=f"anomaly_predictions_{house_view}.csv", mime="text/csv",
        )

with tab3:
    st.subheader(" ¿Qué tan bien detecta cada casa? (sin etiquetas reales)")
    st.caption(
        "No hay ground-truth de anomalías en estos datasets. Para poder medir algo, "
        "se inyectan anomalías sintéticas (una fracción de días del holdout se perturba "
        "sumando N desviaciones estándar a algunas features) y se mide si el ensemble "
        "de esa misma casa las detecta. Esto no mide la detección de anomalías reales, "
        "pero sí qué tan sensible es el modelo de cada casa a desviaciones fuertes "
        "respecto a lo que aprendió como comportamiento normal."
    )

    if not processed_houses:
        st.info(" Selecciona casas y haz click en 'Ejecutar' en la barra lateral.")
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
                title="AUROC frente a anomalías sintéticas (más alto = detector más sensible en esa casa)",
                yaxis_title="AUROC", height=400, yaxis_range=[0, 1.05],
            )
            st.plotly_chart(fig_auroc, use_container_width=True)

        too_small = [h for h in processed_houses if results[h]["synthetic_metrics"] is None]
        if too_small:
            st.warning(
                f"Sin métricas sintéticas para {', '.join(too_small)}: holdout demasiado "
                f"pequeño (menos de 5 días). Baja '% datos para entrenamiento' para dejar "
                f"más días de holdout."
            )

with tab4:
    st.subheader(" Información Técnica")
    st.markdown("""
    ### Métodos del Ensemble

    **Z-Score**: outliers estadísticos univariantes. Score = max(|z|) entre features.

    **Isolation Forest**: aísla anomalías mediante particiones aleatorias en árboles.

    **PCA Reconstruction Error**: proyecta sobre las componentes principales.
    Error = ||x - x̂||²

    **Agregación**: S(x) = w₁·z(x) + w₂·IF(x) + w₃·PCA(x), con cada score
    normalizado usando estadísticas fijadas en el propio `fit()` del detector
    (nunca del batch de `predict()`, para evitar fuga de información).

    ---

    ### Cada casa es independiente

    Cada casa tiene su propio `FeatureScaler` y su propio `EnsembleDetector`,
    entrenados solo con sus propios datos. El proceso es idéntico para todas
    las casas y escala a cuantas se carguen — no hay ninguna casa que
    "entrene" a otra.

    ### Sobre validación

    Estos datasets (CASAS Aruba/Cairo/Milan/Tulum) **no incluyen etiquetas de
    anomalía**. La pestaña "📐 Métricas" usa inyección de anomalías sintéticas
    sobre el holdout de cada casa para poder calcular precision/recall/F1/AUROC
    reales, aunque contra un ground-truth sintético en vez de anomalías
    genuinas del hogar.
    """)

logger.info("Streamlit app ejecutada correctamente")