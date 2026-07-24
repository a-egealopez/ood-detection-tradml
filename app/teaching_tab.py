"""
Pestaña Teaching - Visualización Minimalista y Visual
Cada gráfica dibuja la frontera de decisión real del detector (no una aproximación)
y expone sus propios controles para variar parámetros en tiempo real.

Ubicación: app/teaching_tab.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope as SkEllipticEnvelope
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectors import (
    IsolationForestDetector,
    ExtendedIForestDetector,
    MahalanobisDetector,
    EllipticEnvelopeDetector,
    RobustCovarianceDetector,
    KNNDetector,
    OCSVMDetector,
    LOFDetector,
)
from teaching.datasets import SyntheticDatasetGenerator

GRID_RESOLUTION = 60

METHOD_DESCRIPTIONS: Dict[str, str] = {
    "Isolation Forest": "Aísla puntos con cortes aleatorios: los anómalos quedan aislados en pocos cortes.",
    "Extended IForest": "Como IForest, pero con cortes oblicuos (no solo horizontales/verticales), evitando sesgos direccionales.",
    "Mahalanobis": "Distancia a la media ponderada por la covarianza: mide cuántas elipses de desviación separan al punto.",
    "Elliptic Envelope": "Ajusta una elipse gaussiana a la región 'normal' y marca como anomalía todo lo que queda fuera.",
    "Robust Covariance": "Estima la elipse con MinCovDet, ignorando outliers durante el cálculo (más resistente a contaminación).",
    "KNN": "Score = distancia al k-ésimo vecino más cercano: los puntos aislados de sus vecinos son anómalos.",
    "OC-SVM": "Aprende una frontera no lineal (kernel) que envuelve la región de densidad normal de los datos.",
    "LOF": "Compara la densidad local de un punto contra la de sus vecinos: menor densidad relativa = más anómalo.",
}

DETECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Isolation Forest": {
        "build": lambda p: IsolationForestDetector(contamination=p["contamination"], n_estimators=int(p["n_estimators"])),
        "params": {
            "contamination": ("Contaminación", 0.01, 0.30, 0.05, 0.01),
            "n_estimators": ("N° árboles", 20, 300, 100, 10),
        },
        "family": "iforest",
    },
    "Extended IForest": {
        "build": lambda p: ExtendedIForestDetector(contamination=p["contamination"], n_estimators=int(p["n_estimators"])),
        "params": {
            "contamination": ("Contaminación", 0.01, 0.30, 0.05, 0.01),
            "n_estimators": ("N° árboles", 20, 300, 100, 10),
        },
        "family": "iforest",
    },
    "Mahalanobis": {
        "build": lambda p: MahalanobisDetector(threshold_percentile=p["threshold_percentile"]),
        "params": {
            "threshold_percentile": ("Percentil umbral", 50, 99, 95, 1),
        },
        "family": "covariance_empirical",
    },
    "Elliptic Envelope": {
        "build": lambda p: EllipticEnvelopeDetector(contamination=p["contamination"]),
        "params": {
            "contamination": ("Contaminación", 0.01, 0.30, 0.05, 0.01),
        },
        "family": "covariance_elliptic",
    },
    "Robust Covariance": {
        "build": lambda p: RobustCovarianceDetector(contamination=p["contamination"]),
        "params": {
            "contamination": ("Contaminación", 0.01, 0.30, 0.05, 0.01),
        },
        "family": "covariance_robust",
    },
    "KNN": {
        "build": lambda p: KNNDetector(n_neighbors=int(p["n_neighbors"])),
        "params": {
            "n_neighbors": ("K vecinos", 2, 50, 5, 1),
        },
        "family": "knn",
    },
    "OC-SVM": {
        "build": lambda p: OCSVMDetector(nu=p["nu"]),
        "params": {
            "nu": ("Nu (fracción outliers)", 0.01, 0.5, 0.05, 0.01),
        },
        "family": "boundary_only",
    },
    "LOF": {
        "build": lambda p: LOFDetector(n_neighbors=int(p["n_neighbors"])),
        "params": {
            "n_neighbors": ("K vecinos", 5, 60, 20, 1),
        },
        "family": "lof",
    },
}


def _score_grid(detector: Any, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corre el detector ya entrenado sobre una malla para dibujar su frontera de decisión real."""
    xs = np.linspace(x_range[0], x_range[1], GRID_RESOLUTION)
    ys = np.linspace(y_range[0], y_range[1], GRID_RESOLUTION)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    _, grid_scores = detector.predict(grid_points)
    return xx, yy, grid_scores.reshape(xx.shape)


def _ellipse_points(mean: np.ndarray, cov: np.ndarray, n_std: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.linspace(0, 2 * np.pi, n_points)
    axis_lengths = n_std * np.sqrt(np.clip(eigvals, 0, None))
    circle = np.stack([np.cos(angle), np.sin(angle)]) * axis_lengths[:, None]
    ellipse = eigvecs @ circle
    return ellipse[0] + mean[0], ellipse[1] + mean[1]


def _add_ellipse_traces(fig: go.Figure, mean: np.ndarray, cov: np.ndarray) -> None:
    for n_std, color in [(1, "rgba(46,139,87,0.9)"), (2, "rgba(218,165,32,0.9)"), (3, "rgba(178,34,34,0.9)")]:
        ex, ey = _ellipse_points(mean, cov, n_std)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))


def _add_covariance_overlay(fig: go.Figure, X: np.ndarray, family: str) -> None:
    try:
        if family == "covariance_empirical":
            model = EmpiricalCovariance().fit(X)
            _add_ellipse_traces(fig, model.location_, model.covariance_)
        elif family == "covariance_robust":
            model = MinCovDet(random_state=42).fit(X)
            _add_ellipse_traces(fig, model.location_, model.covariance_)
        elif family == "covariance_elliptic":
            model = SkEllipticEnvelope(random_state=42).fit(X)
            _add_ellipse_traces(fig, model.location_, model.covariance_)
    except Exception:
        pass


def _add_knn_illustration(fig: go.Figure, X: np.ndarray, scores: np.ndarray, k: int) -> None:
    """Traza líneas del punto más anómalo hacia sus k vecinos más cercanos usados para su score."""
    target_idx = int(np.argmax(scores))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, neighbor_idx = nn.kneighbors(X[target_idx].reshape(1, -1))
    for idx in neighbor_idx[0][1:]:
        fig.add_trace(go.Scatter(
            x=[X[target_idx, 0], X[idx, 0]], y=[X[target_idx, 1], X[idx, 1]],
            mode="lines", line=dict(color="rgba(178,34,34,0.6)", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[X[target_idx, 0]], y=[X[target_idx, 1]], mode="markers",
        marker=dict(size=13, color="gold", line=dict(width=2, color="black"), symbol="star"),
        hoverinfo="skip", showlegend=False,
    ))


def _add_lof_illustration(fig: go.Figure, X: np.ndarray, k: int, sample_size: int = 12) -> None:
    """Círculos con radio al k-ésimo vecino de una muestra de puntos: ilustra densidad local variable."""
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    distances, _ = nn.kneighbors(X)
    theta = np.linspace(0, 2 * np.pi, 40)
    for idx in sample_idx:
        radius = distances[idx, -1]
        fig.add_trace(go.Scatter(
            x=X[idx, 0] + radius * np.cos(theta), y=X[idx, 1] + radius * np.sin(theta),
            mode="lines", line=dict(color="rgba(70,130,180,0.5)", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))


def _build_figure(
    detector_name: str, detector: Any, X: np.ndarray,
    y_pred: np.ndarray, scores: np.ndarray, auroc: float, k_for_illustration: int,
) -> go.Figure:
    fig = go.Figure()

    x_margin = 0.1 * (X[:, 0].max() - X[:, 0].min() + 1e-6)
    y_margin = 0.1 * (X[:, 1].max() - X[:, 1].min() + 1e-6)
    x_range = (X[:, 0].min() - x_margin, X[:, 0].max() + x_margin)
    y_range = (X[:, 1].min() - y_margin, X[:, 1].max() + y_margin)

    xx, yy, zz = _score_grid(detector, x_range, y_range)
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=zz,
        colorscale="Blues", showscale=False, opacity=0.55,
        contours=dict(showlines=False),
        hoverinfo="skip",
    ))

    family = DETECTOR_REGISTRY[detector_name]["family"]
    if family in ("covariance_empirical", "covariance_robust", "covariance_elliptic"):
        _add_covariance_overlay(fig, X, family)
    elif family == "knn":
        _add_knn_illustration(fig, X, scores, k_for_illustration)
    elif family == "lof":
        _add_lof_illustration(fig, X, k_for_illustration)

    normal_mask = y_pred == 0
    fig.add_trace(go.Scatter(
        x=X[normal_mask, 0], y=X[normal_mask, 1], mode="markers",
        marker=dict(size=5, color="rgba(70,130,180,0.6)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    anomaly_mask = y_pred == 1
    if anomaly_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=X[anomaly_mask, 0], y=X[anomaly_mask, 1], mode="markers",
            marker=dict(size=7, color=scores[anomaly_mask], colorscale="Reds", showscale=False, cmin=0, cmax=1, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"<b>{detector_name}</b>", font=dict(size=18, color="#1a1a1a"), x=0.5, xanchor="center"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=x_range),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_range),
        hovermode=False,
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        template="plotly_white",
        font=dict(color="#1a1a1a"),
        paper_bgcolor="rgba(248,249,250,1)",
        plot_bgcolor="rgba(248,249,250,1)",
    )
    return fig


def _fit_and_score(detector_name: str, params: Dict[str, float], X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    detector = DETECTOR_REGISTRY[detector_name]["build"](params)
    detector.fit(X)
    y_pred, scores = detector.predict(X)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = 0.0
    return {"detector": detector, "y_pred": y_pred, "scores": scores, "auroc": auroc}


def render_teaching_tab() -> None:
    st.markdown("""
    ## 🎓 Learning: How Anomaly Detectors Work

    Elige la geometría del dataset y observa cómo cada algoritmo dibuja su propia frontera de decisión.
    El fondo azul es el score de anomalía que el propio detector asigna a cada región del espacio,
    no una simulación: es la salida real de `detector.predict()` evaluada sobre una malla.
    """)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        dataset_key = st.selectbox(
            "Select a dataset:",
            options=list(SyntheticDatasetGenerator.DATASETS.keys()),
            label_visibility="collapsed",
        )
    with col2:
        n_samples = st.slider("Samples", 100, 500, 300, step=50, label_visibility="collapsed")
    with col3:
        contamination_global = st.slider("Anomalies %", 5, 30, 15, step=1, label_visibility="collapsed") / 100.0

    dataset_key_internal = SyntheticDatasetGenerator.DATASETS[dataset_key]
    X, y_true, dataset_description = SyntheticDatasetGenerator.generate(
        dataset_key_internal, n_samples=n_samples, contamination=contamination_global, random_state=42,
    )
    st.markdown(f"**{dataset_description}**")
    st.markdown("---")
    st.markdown("### All Detectors at a Glance")

    dataset_signature = (dataset_key_internal, n_samples, round(contamination_global, 4))
    detector_names = list(DETECTOR_REGISTRY.keys())

    for row_start in range(0, len(detector_names), 2):
        cols = st.columns(2, gap="medium")
        for col_offset, col in enumerate(cols):
            idx = row_start + col_offset
            if idx >= len(detector_names):
                break
            detector_name = detector_names[idx]
            spec = DETECTOR_REGISTRY[detector_name]

            with col:
                params: Dict[str, float] = {}
                for param_name, (_, _, _, p_default, _) in spec["params"].items():
                    widget_key = f"slider_{detector_name}_{param_name}_{dataset_key_internal}_{n_samples}"
                    params[param_name] = st.session_state.get(widget_key, p_default)

                result_signature = (detector_name, dataset_signature, tuple(sorted(params.items())))
                cache_key = f"cache_{detector_name}"
                if st.session_state.get(f"{cache_key}_sig") != result_signature:
                    st.session_state[cache_key] = _fit_and_score(detector_name, params, X, y_true)
                    st.session_state[f"{cache_key}_sig"] = result_signature
                result = st.session_state[cache_key]

                k_illustration = int(params.get("n_neighbors", 5))
                fig = _build_figure(
                    detector_name, result["detector"], X,
                    result["y_pred"], result["scores"], result["auroc"], k_illustration,
                )
                st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
                st.caption(f"AUROC: {result['auroc']:.3f} — {METHOD_DESCRIPTIONS.get(detector_name, '')}")

                for param_name, (label, p_min, p_max, p_default, p_step) in spec["params"].items():
                    widget_key = f"slider_{detector_name}_{param_name}_{dataset_key_internal}_{n_samples}"
                    value_type = type(p_default)
                    st.slider(
                        label, value_type(p_min), value_type(p_max), value_type(params[param_name]),
                        step=value_type(p_step), key=widget_key,
                    )

    st.markdown("---")
    st.markdown("### Detector Ranking by AUROC")

    ranking = sorted(
        ((name, st.session_state[f"cache_{name}"]) for name in detector_names if f"cache_{name}" in st.session_state),
        key=lambda item: item[1]["auroc"], reverse=True,
    )
    ranking_data = {
        "Rank": [f"#{i + 1}" for i in range(len(ranking))],
        "Detector": [name for name, _ in ranking],
        "AUROC": [f"{r['auroc']:.3f}" for _, r in ranking],
    }
    _, mid, _ = st.columns([0.5, 2, 0.5])
    with mid:
        st.dataframe(ranking_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("💡 Understanding the Visualizations"):
        st.markdown("""
        **Fondo azul** = score de anomalía real del detector, evaluado sobre toda la malla del espacio.
        **Puntos azules** = Normal (predicho) · **Puntos rojos** = Anomalía (predicho), intensidad = confianza.
        **Elipses punteadas** (Mahalanobis / Elliptic Envelope / Robust Covariance) = contornos de 1σ, 2σ y 3σ
        de la covarianza estimada; los puntos fuera de la elipse externa son los más atípicos.
        **Estrella dorada + líneas rojas** (KNN) = el punto más anómalo y los k vecinos usados para calcular su score.
        **Círculos azules** (LOF) = radio al k-ésimo vecino de una muestra de puntos, mostrando cómo varía
        la densidad local de una zona a otra del dataset.

        **AUROC** = Area Under ROC Curve · 1.0 separación perfecta · 0.5 azar · > 0.7 buen detector para esta geometría.

        Mueve los deslizadores bajo cada gráfica: el detector se re-entrena al instante con el nuevo parámetro.
        """)

    with st.expander("🔧 Dataset Information"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Points", f"{len(X):,}")
        c2.metric("Normal", f"{(y_true == 0).sum():,}")
        c3.metric("Anomalies", f"{(y_true == 1).sum():,}")
        c4.metric("Dimensions", "2D")


if __name__ == "__main__":
    render_teaching_tab()