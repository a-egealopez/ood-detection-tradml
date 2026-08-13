"""Teaching track: visualize each detector's real decision boundary on 2-D data.

Every chart draws the actual anomaly-score field of the fitted detector over a mesh
(no approximations) and exposes per-detector sliders that retrain it in real time.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.covariance import EllipticEnvelope as SkEllipticEnvelope
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import metric_row, section_title  # noqa: E402
from detectors.factory import build_detector  # noqa: E402
from streamlit_config import DETECTOR_REGISTRY  # noqa: E402
from teaching.datasets import SyntheticDatasetGenerator  # noqa: E402
from theme import (  # noqa: E402
    ANOMALY_SOFT,
    BACKGROUND,
    PRIMARY_SOFT,
    TEXT,
    display_chart,
)

GRID_RESOLUTION = 60

# Detectors suited to 2-D teaching grids (sequential models are excluded).
TEACHING_DETECTORS = [
    name
    for name, spec in DETECTOR_REGISTRY.items()
    if spec.category != "Sequential"
]


def _score_grid(
    detector: Any, x_range: tuple[float, float], y_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a fitted detector over a mesh to draw its real decision boundary."""
    xs = np.linspace(x_range[0], x_range[1], GRID_RESOLUTION)
    ys = np.linspace(y_range[0], y_range[1], GRID_RESOLUTION)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    _, grid_scores = detector.predict(grid_points)
    return xx, yy, grid_scores.reshape(xx.shape)


def _ellipse_points(
    mean: np.ndarray, cov: np.ndarray, n_std: float, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.linspace(0, 2 * np.pi, n_points)
    axis_lengths = n_std * np.sqrt(np.clip(eigvals, 0, None))
    circle = np.stack([np.cos(angle), np.sin(angle)]) * axis_lengths[:, None]
    ellipse = eigvecs @ circle
    return ellipse[0] + mean[0], ellipse[1] + mean[1]


def _add_ellipse_traces(fig: go.Figure, mean: np.ndarray, cov: np.ndarray) -> None:
    for n_std, color in [
        (1, PRIMARY_SOFT),
        (2, "rgba(100,116,139,0.7)"),
        (3, ANOMALY_SOFT),
    ]:
        ex, ey = _ellipse_points(mean, cov, n_std)
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line={"color": color, "width": 2, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            )
        )


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


def _add_knn_illustration(
    fig: go.Figure, X: np.ndarray, scores: np.ndarray, k: int
) -> None:
    """Draw lines from the most anomalous point to the k neighbors used for its score."""
    target_idx = int(np.argmax(scores))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, neighbor_idx = nn.kneighbors(X[target_idx].reshape(1, -1))
    for idx in neighbor_idx[0][1:]:
        fig.add_trace(
            go.Scatter(
                x=[X[target_idx, 0], X[idx, 0]],
                y=[X[target_idx, 1], X[idx, 1]],
                mode="lines",
                line={"color": ANOMALY_SOFT, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[X[target_idx, 0]],
            y=[X[target_idx, 1]],
            mode="markers",
            marker={
                "size": 13,
                "color": "gold",
                "line": {"width": 2, "color": "black"},
                "symbol": "star",
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_lof_illustration(
    fig: go.Figure, X: np.ndarray, k: int, sample_size: int = 12
) -> None:
    """Circles of radius = distance to the k-th neighbor illustrate local density."""
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    distances, _ = nn.kneighbors(X)
    theta = np.linspace(0, 2 * np.pi, 40)
    for idx in sample_idx:
        radius = distances[idx, -1]
        fig.add_trace(
            go.Scatter(
                x=X[idx, 0] + radius * np.cos(theta),
                y=X[idx, 1] + radius * np.sin(theta),
                mode="lines",
                line={"color": PRIMARY_SOFT, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _build_figure(
    detector_name: str,
    detector: Any,
    X: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    k_for_illustration: int,
) -> go.Figure:
    family = DETECTOR_REGISTRY[detector_name].family

    x_margin = 0.1 * (X[:, 0].max() - X[:, 0].min() + 1e-6)
    y_margin = 0.1 * (X[:, 1].max() - X[:, 1].min() + 1e-6)
    x_range = (X[:, 0].min() - x_margin, X[:, 0].max() + x_margin)
    y_range = (X[:, 1].min() - y_margin, X[:, 1].max() + y_margin)

    fig = go.Figure()

    xx, yy, zz = _score_grid(detector, x_range, y_range)
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale="Blues",
            showscale=False,
            opacity=0.55,
            contours={"showlines": False},
            hoverinfo="skip",
        )
    )

    if family in ("covariance_empirical", "covariance_robust", "covariance_elliptic"):
        _add_covariance_overlay(fig, X, family)
    elif family == "knn":
        _add_knn_illustration(fig, X, scores, k_for_illustration)
    elif family == "lof":
        _add_lof_illustration(fig, X, k_for_illustration)

    normal_mask = y_pred == 0
    fig.add_trace(
        go.Scatter(
            x=X[normal_mask, 0],
            y=X[normal_mask, 1],
            mode="markers",
            marker={
                "size": 5,
                "color": PRIMARY_SOFT,
                "line": {"width": 0},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    anomaly_mask = y_pred == 1
    if anomaly_mask.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=X[anomaly_mask, 0],
                y=X[anomaly_mask, 1],
                mode="markers",
                marker={
                    "size": 7,
                    "color": scores[anomaly_mask],
                    "colorscale": "Reds",
                    "showscale": False,
                    "cmin": 0,
                    "cmax": 1,
                    "line": {"width": 0},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis={"showgrid": False, "zeroline": False, "visible": False, "range": x_range},
        yaxis={"showgrid": False, "zeroline": False, "visible": False, "range": y_range},
        hovermode=False,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    fig.update_layout(
        title={
            "text": f"<b>{detector_name}</b>",
            "font": {"size": 18, "color": TEXT},
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        font={"color": TEXT},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        height=380,
    )
    return fig


def _fit_and_score(
    detector_name: str, params: dict[str, float], X: np.ndarray, y_true: np.ndarray
) -> dict[str, Any]:
    detector = build_detector(detector_name, params)
    detector.fit(X)
    y_pred, scores = detector.predict(X)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = 0.0
    return {"detector": detector, "y_pred": y_pred, "scores": scores, "auroc": auroc}


def _render_detector_grid(X: np.ndarray, y_true: np.ndarray, dataset_key: str, n_samples: int) -> None:
    dataset_signature = (dataset_key, n_samples, int((y_true == 1).sum()))

    for row_start in range(0, len(TEACHING_DETECTORS), 2):
        cols = st.columns(2, gap="medium")
        for col_offset, col in enumerate(cols):
            idx = row_start + col_offset
            if idx >= len(TEACHING_DETECTORS):
                break
            detector_name = TEACHING_DETECTORS[idx]
            spec = DETECTOR_REGISTRY[detector_name]

            with col:
                params: dict[str, float] = {}
                for param in spec.params:
                    widget_key = f"slider_{detector_name}_{param.kwarg}_{dataset_key}_{n_samples}"
                    params[param.kwarg] = st.session_state.get(widget_key, param.default)

                result_signature = (
                    detector_name,
                    dataset_signature,
                    tuple(sorted(params.items())),
                )
                cache_key = f"cache_{detector_name}"
                if st.session_state.get(f"{cache_key}_sig") != result_signature:
                    st.session_state[cache_key] = _fit_and_score(
                        detector_name, params, X, y_true
                    )
                    st.session_state[f"{cache_key}_sig"] = result_signature
                result = st.session_state[cache_key]

                k_illustration = int(params.get("n_neighbors", 5))
                fig = _build_figure(
                    detector_name,
                    result["detector"],
                    X,
                    result["y_pred"],
                    result["scores"],
                    k_illustration,
                )
                display_chart(fig, key=f"chart_{detector_name}_{dataset_key}_{n_samples}")

                st.caption(
                    f"AUROC: {result['auroc']:.3f} - {spec.description}"
                )

                for param in spec.params:
                    widget_key = f"slider_{detector_name}_{param.kwarg}_{dataset_key}_{n_samples}"
                    value_type = type(param.default)
                    st.slider(
                        param.label,
                        value_type(param.min),
                        value_type(param.max),
                        value_type(params[param.kwarg]),
                        step=value_type(param.step),
                        key=widget_key,
                    )


def render_teaching_view() -> None:
    st.markdown(
        """
        ## Learning: How Anomaly Detectors Work

        Pick a dataset geometry and observe how each algorithm draws its own decision
        boundary. The blue background is the anomaly score the detector itself assigns
        to every region of the space — the real output of `detector.predict()` evaluated
        on a mesh, not an approximation.
        """
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        dataset_key = st.selectbox(
            "Select a dataset:",
            options=list(SyntheticDatasetGenerator.DATASETS.keys()),
            label_visibility="collapsed",
        )
    with col2:
        n_samples = st.slider(
            "Samples", 100, 500, 300, step=50, label_visibility="collapsed"
        )
    with col3:
        contamination_global = (
            st.slider("Anomalies %", 5, 30, 15, step=1, label_visibility="collapsed")
            / 100.0
        )

    dataset_key_internal = SyntheticDatasetGenerator.DATASETS[dataset_key]
    X, y_true, dataset_description = SyntheticDatasetGenerator.generate(
        dataset_key_internal,
        n_samples=n_samples,
        contamination=contamination_global,
        random_state=42,
    )
    st.markdown(f"**{dataset_description}**")

    st.markdown("---")
    section_title("All Detectors at a Glance")
    _render_detector_grid(X, y_true, dataset_key_internal, n_samples)

    st.markdown("---")
    section_title("Detector Ranking by AUROC")

    ranking = sorted(
        (
            (name, st.session_state[f"cache_{name}"])
            for name in TEACHING_DETECTORS
            if f"cache_{name}" in st.session_state
        ),
        key=lambda item: item[1]["auroc"],
        reverse=True,
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
    with st.expander("Understanding the Visualizations"):
        st.markdown(
            """
            **Blue background** = the detector's real anomaly score, evaluated over the whole space.
            **Blue points** = predicted normal - **Red points** = predicted anomaly, intensity = confidence.
            **Dotted ellipses** (Mahalanobis / Elliptic Envelope / Robust Covariance) = 1σ, 2σ and 3σ
            contours of the estimated covariance; points outside the outer ellipse are the most atypical.
            **Gold star + red lines** (KNN) = the most anomalous point and the k neighbors used to score it.
            **Blue circles** (LOF) = radius to the k-th neighbor for a sample of points, showing how the
            local density varies across the dataset.

            **AUROC** = Area Under the ROC Curve - 1.0 perfect separation - 0.5 random - above 0.7 is a
            strong detector for this geometry.

            Move the sliders under each chart: the detector is retrained instantly with the new parameter.
            """
        )

    with st.expander("Dataset Information"):
        metric_row(
            [
                ("Total Points", f"{len(X):,}"),
                ("Normal", f"{(y_true == 0).sum():,}"),
                ("Anomalies", f"{(y_true == 1).sum():,}"),
                ("Dimensions", "2D"),
            ]
        )
