"""
feature_extraction_tab.py

Pestaña didáctica: cómo se extraen los vectores de características a partir de un
stream de eventos (sensores IoT), tanto sobre datos sintéticos como sobre el dataset
real CASAS Aruba. Permite elegir entre 3 métodos de la literatura de event-driven
time series y ver, para un día concreto, el cálculo paso a paso.

Ubicación: app/feature_extraction_tab.py
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.sqlite_manager import SQLiteDataManager
from config import DB_PATH
from features.event_driven_extractors import (
    WindowAggregationExtractor,
    IntervalStatisticsExtractor,
    NGramTransitionExtractor,
    generate_synthetic_events,
)

TEXT_COLOR = "#1a1a1a"
BG_COLOR = "rgba(248,249,250,1)"

METHOD_DESCRIPTIONS = {
    "Window Aggregation": (
        "Agrupa los eventos de cada día y resume su distribución horaria en estadísticas "
        "(conteo, entropía, % nocturno). Es el enfoque clásico en reconocimiento de actividades "
        "sobre sensores binarios."
    ),
    "Inter-Event Interval (IEI)": (
        "Analiza el tiempo entre eventos consecutivos como un point process. El coeficiente de "
        "variación (CV) y el Fano factor distinguen procesos regulares, tipo Poisson o 'bursty'."
    ),
    "N-gram Transition (Markov)": (
        "Convierte la secuencia de sensores disparados en una cadena de Markov de primer orden. "
        "La entropía de la matriz de transición mide qué tan predecible es la secuencia de eventos."
    ),
}

METHOD_ANOMALY_TYPES = {
    "Window Aggregation": (
        "Contextual + Colectiva (día completo)",
        "El contexto (hora/franja) está codificado en las features (ej. `night_activity_ratio`), "
        "y cada vector resume un día entero: detecta días atípicos como conjunto, no eventos sueltos.",
    ),
    "Inter-Event Interval (IEI)": (
        "Puntual (crudo) o Colectiva (agregado)",
        "Los intervalos crudos sirven para marcar un único hueco anómalo (puntual). El vector que "
        "se extrae aquí está agregado por día, así que detecta días con un 'ritmo' global atípico "
        "(colectiva), no el instante exacto del hueco.",
    ),
    "N-gram Transition (Markov)": (
        "Colectiva de secuencia (pattern-based)",
        "Solo mira el orden de los eventos, no su instante ni magnitud: detecta rutinas que rompen "
        "el patrón habitual de secuencia, aunque cada sensor e intervalo sean normales por separado.",
    ),
}


@st.cache_data
def _load_real_data() -> pd.DataFrame:
    db = SQLiteDataManager(str(DB_PATH))
    db.connect()
    df = db.query_to_dataframe("SELECT * FROM sensor_events ORDER BY timestamp")
    db.close()
    return df


def _base_layout(fig: go.Figure, title: str, height: int = 340) -> None:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=TEXT_COLOR), x=0.5, xanchor="center"),
        height=height,
        margin=dict(l=40, r=20, t=45, b=40),
        showlegend=False,
        template="plotly_white",
        font=dict(color=TEXT_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
    )


def _plot_window_aggregation(group: pd.DataFrame, diag: dict) -> Tuple[go.Figure, go.Figure]:
    group = group.copy()
    group["timestamp"] = pd.to_datetime(group["timestamp"])
    sensors = sorted(group["sensor_id"].unique())

    timeline = go.Figure()
    for sensor in sensors:
        sub = group[group["sensor_id"] == sensor]
        timeline.add_trace(go.Scatter(
            x=sub["timestamp"], y=[sensor] * len(sub), mode="markers",
            marker=dict(size=8, color="rgba(70,130,180,0.75)"),
            hoverinfo="skip",
        ))
    timeline.add_vrect(
        x0=group["timestamp"].dt.normalize().iloc[0],
        x1=group["timestamp"].dt.normalize().iloc[0] + pd.Timedelta(hours=8),
        fillcolor="rgba(178,34,34,0.08)", line_width=0,
    )
    timeline.add_vrect(
        x0=group["timestamp"].dt.normalize().iloc[0] + pd.Timedelta(hours=22),
        x1=group["timestamp"].dt.normalize().iloc[0] + pd.Timedelta(hours=24),
        fillcolor="rgba(178,34,34,0.08)", line_width=0,
    )
    _base_layout(timeline, "Eventos del día (rojo = tramo nocturno 22h-8h)")
    timeline.update_xaxes(title_text="Hora")
    timeline.update_yaxes(title_text="Sensor")

    hourly = diag["hourly_counts"]
    hist = go.Figure(go.Bar(x=list(hourly.index), y=hourly.values, marker_color="rgba(70,130,180,0.8)"))
    _base_layout(hist, "Distribución horaria (base de la entropía)")
    hist.update_xaxes(title_text="Hora del día")
    hist.update_yaxes(title_text="N° eventos")

    return timeline, hist


def _plot_interval_statistics(group: pd.DataFrame, diag: dict) -> Tuple[go.Figure, go.Figure]:
    group = group.copy()
    group["timestamp"] = pd.to_datetime(group["timestamp"]).sort_values()
    ts_sorted = group.sort_values("timestamp")["timestamp"]

    timeline = go.Figure()
    timeline.add_trace(go.Scatter(
        x=ts_sorted, y=[0] * len(ts_sorted), mode="markers",
        marker=dict(size=9, color="rgba(70,130,180,0.8)", symbol="line-ns", line=dict(width=2, color="rgba(70,130,180,0.8)")),
        hoverinfo="skip",
    ))
    sample_pairs = list(zip(ts_sorted.iloc[:-1], ts_sorted.iloc[1:]))[:6]
    for t0, t1 in sample_pairs:
        timeline.add_shape(
            type="line", x0=t0, x1=t1, y0=0.15, y1=0.15,
            line=dict(color="rgba(178,34,34,0.7)", width=1.5, dash="dot"),
        )
    _base_layout(timeline, "Eventos del día (líneas rojas = intervalos ilustrados)")
    timeline.update_xaxes(title_text="Hora")
    timeline.update_yaxes(visible=False, range=[-1, 1])

    intervals_min = diag["intervals_seconds"] / 60.0
    hist = go.Figure(go.Histogram(x=intervals_min, nbinsx=25, marker_color="rgba(70,130,180,0.8)"))
    _base_layout(hist, "Distribución de intervalos entre eventos (IEI)")
    hist.update_xaxes(title_text="Minutos entre eventos consecutivos")
    hist.update_yaxes(title_text="Frecuencia")

    return timeline, hist


def _plot_ngram_transition(diag: dict) -> Tuple[go.Figure, go.Figure]:
    sequence = diag["sequence"]
    tokens = sequence[:40]

    strip = go.Figure()
    strip.add_trace(go.Scatter(
        x=list(range(len(tokens))), y=[0] * len(tokens), mode="markers+text",
        marker=dict(size=14, color="rgba(70,130,180,0.8)"),
        text=tokens, textposition="top center", textfont=dict(size=9, color=TEXT_COLOR),
        hoverinfo="skip",
    ))
    for i in range(len(tokens) - 1):
        strip.add_annotation(
            x=i + 1, y=0, ax=i, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="rgba(178,34,34,0.6)",
        )
    _base_layout(strip, "Secuencia de sensores disparados (primeros 40 eventos)", height=260)
    strip.update_xaxes(visible=False)
    strip.update_yaxes(visible=False, range=[-1, 1])

    matrix = diag["transition_matrix"]
    heatmap = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
        colorscale="Blues", showscale=False,
    ))
    _base_layout(heatmap, "Matriz de transición (bigramas sensor A → sensor B)")

    return strip, heatmap


def render_feature_extraction_tab() -> None:
    st.markdown("""
    ## 🧬 Feature Extraction: del stream de eventos al vector de características

    Elige el origen de los datos y el método de extracción para ver, paso a paso,
    cómo un conjunto de eventos crudos (timestamp, sensor, tipo) se convierte en el
    vector numérico que después consumen los detectores de anomalías.
    """)

    col_source, col_method = st.columns([1, 1.4])
    with col_source:
        data_source = st.radio("Origen de los datos", ["Sintético", "Real (CASAS Aruba)"], horizontal=True)
    with col_method:
        method_name = st.radio("Método de extracción", list(METHOD_DESCRIPTIONS.keys()), horizontal=True)

    st.caption(METHOD_DESCRIPTIONS[method_name])
    anomaly_type, anomaly_rationale = METHOD_ANOMALY_TYPES[method_name]
    st.info(f"**Tipo de anomalía que detecta: {anomaly_type}.** {anomaly_rationale}")

    if data_source == "Sintético":
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            pattern = st.selectbox("Patrón temporal", ["regular", "bursty", "day_night"])
        with s2:
            n_days = st.slider("N° de días", 2, 10, 4)
        with s3:
            n_sensors = st.slider("N° de sensores", 2, 6, 3)
        with s4:
            events_per_day = st.slider("Eventos/día", 20, 200, 80, step=10)

        df = generate_synthetic_events(
            n_days=n_days, pattern=pattern, n_sensors=n_sensors,
            events_per_day=events_per_day, seed=42,
        )
    else:
        try:
            df = _load_real_data()
        except Exception as e:
            st.error(f"No se pudo cargar la base de datos: {e}")
            st.stop()
        if df.empty:
            st.warning("La base de datos está vacía. Corre primero `python src/ingestion/casas_loader.py`.")
            st.stop()
        max_days = st.slider("N° de días a analizar (desde el inicio del dataset)", 3, 30, 10)
        df = df[pd.to_datetime(df["timestamp"]).dt.date.isin(
            sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())[:max_days]
        )]

    st.markdown("---")

    if method_name == "Window Aggregation":
        extractor = WindowAggregationExtractor()
    elif method_name == "Inter-Event Interval (IEI)":
        extractor = IntervalStatisticsExtractor()
    else:
        extractor = NGramTransitionExtractor()
        extractor.fit_vocabulary(df)

    X, dates = extractor.extract(df)
    df_dates = pd.to_datetime(df["timestamp"]).dt.date

    selected_day = st.selectbox("Día a inspeccionar en detalle", list(dates))
    group = df[df_dates == selected_day]

    diag = extractor.diagnostics(group)

    if method_name == "Window Aggregation":
        fig_a, fig_b = _plot_window_aggregation(group, diag)
    elif method_name == "Inter-Event Interval (IEI)":
        fig_a, fig_b = _plot_interval_statistics(group, diag)
    else:
        fig_a, fig_b = _plot_ngram_transition(diag)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_a, use_container_width=True, theme=None, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(fig_b, use_container_width=True, theme=None, config={"displayModeBar": False})

    st.markdown(f"#### Vector de características extraído — {selected_day}")
    feature_row = pd.DataFrame([diag["features"]], columns=diag["feature_names"])
    st.dataframe(feature_row, use_container_width=True, hide_index=True)

    st.markdown("#### Vectores extraídos para todos los días del dataset")
    full_table = pd.DataFrame(X, columns=extractor.FEATURE_NAMES)
    full_table.insert(0, "date", dates)
    st.dataframe(full_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_feature_extraction_tab()
