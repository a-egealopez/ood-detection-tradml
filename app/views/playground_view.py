"""2D Playground: visualize each detector's real decision boundary on 2-D data.

Every chart draws the actual anomaly-score field of the fitted detector over a mesh
(no approximations) and exposes per-detector sliders that retrain it in real time.
Detectors are grouped by family (Gaussian, Density, Distance, One-Class SVM, ...) so the grid
reads as a visual taxonomy instead of a flat list.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import (  # noqa: E402
    breadcrumb,
    detector_card,
    family_header,
    metric_row,
    read_param_values,
    section_title,
)
from mesh import score_mesh  # noqa: E402
from streamlit_config import DETECTOR_REGISTRY  # noqa: E402
from theme import (  # noqa: E402
    ANOMALY_SOFT,
    BG_CANVAS,
    BG_SURFACE,
    BORDER,
    PRIMARY_SOFT,
    TEXT,
)

from detectors.factory import build_detector  # noqa: E402
from teaching.datasets import SyntheticDatasetGenerator  # noqa: E402

# Detectors suited to 2-D teaching grids (sequential models are excluded).
TEACHING_DETECTORS = [
    name for name, spec in DETECTOR_REGISTRY.items() if spec.category != "Sequential"
]

# One-line didactic hook per detector family, shown under each section header.
CATEGORY_CAPTIONS = {
    "Density": "Detectors that isolate low-density regions of the space.",
    "Gaussian": "Detectors that model the normal data as a Gaussian distribution (empirical or robust covariance).",
    "Distance": "Detectors that score points by distance to their nearest training neighbors.",
    "One-Class SVM": "One-class boundary learning with kernels.",
    "Univariate": "Per-feature standard deviation from the training mean.",
    "Dimensionality": "Reconstruction error to the dominant linear subspace.",
}


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


def _add_covariance_overlay(fig: go.Figure, detector: Any, family: str) -> None:
    """Draw the 1/2/3-sigma ellipses of the covariance THE DETECTOR fits.

    Uses the fitted parameters stored on the detector itself (not a fresh refit),
    so the ellipses match the detector's real decision surface.
    """
    try:
        if family == "covariance_empirical":
            mean, cov = detector.mean, detector.cov
        elif family in ("covariance_robust", "covariance_elliptic"):
            mean = detector.model.location_
            cov = detector.model.covariance_
        else:
            return
        _add_ellipse_traces(fig, mean, cov)
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw covariance overlay for %s", family)


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


def _add_ocsvm_boundary(
    fig: go.Figure, detector: Any, x_range: tuple[float, float], y_range: tuple[float, float]
) -> None:
    """Draw the One-Class SVM decision boundary (decision_function = 0).

    The colored mesh already shows the score field; this contour line makes the
    exact frontier crisp, which is the thing an SVM boundary learner is about.
    """
    try:
        grid = 60
        xs = np.linspace(x_range[0], x_range[1], grid)
        ys = np.linspace(y_range[0], y_range[1], grid)
        xx, yy = np.meshgrid(xs, ys)
        mesh = np.column_stack([xx.ravel(), yy.ravel()])
        decision = detector.model.decision_function(mesh).reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=decision,
                contours={"start": 0, "end": 0, "size": 1, "coloring": "lines"},
                line={"color": "white", "width": 2.5},
                showscale=False,
                hoverinfo="skip",
            )
        )
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw OC-SVM boundary overlay")


def _add_zscore_band(fig: go.Figure, detector: Any) -> None:
    """Draw the axis-aligned threshold rectangle mu +/- threshold*sigma per feature.

    Z-Score is univariate: each feature is judged independently, so its "normal"
    region is an axis-parallel box, not a rotated ellipse. This rectangle makes
    that explicit.
    """
    try:
        mu, sigma, t = detector.mu, detector.sigma, detector.threshold
        x0, x1 = mu[0] - t * sigma[0], mu[0] + t * sigma[0]
        y0, y1 = mu[1] - t * sigma[1], mu[1] + t * sigma[1]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                line={"color": "white", "width": 2.5, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw Z-Score band overlay")


def _add_pca_axes(fig: go.Figure, detector: Any, X: np.ndarray) -> None:
    """Draw the principal-component directions from the mean.

    PCA Reconstruction scores a point by distance to the subspace spanned by the
    top components; showing those directions on the plane explains why points off
    the main manifold get high anomaly scores.
    """
    try:
        W = np.asarray(detector.W)
        mean = np.asarray(detector.mean)
        span = X.max(axis=0) - X.min(axis=0)
        length = 0.8 * np.max(span)
        for col in range(W.shape[1]):
            direction = W[:, col]
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            endpoint = mean + direction * length
            fig.add_trace(
                go.Scatter(
                    x=[mean[0], endpoint[0]],
                    y=[mean[1], endpoint[1]],
                    mode="lines",
                    line={"color": "white", "width": 2.5},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw PCA axes overlay")


def _add_iforest_iso(
    fig: go.Figure, detector: Any, x_range: tuple[float, float], y_range: tuple[float, float]
) -> None:
    """Overlay iso-lines of the isolation score to reveal the 'isolation valleys'.

    The mesh is the score field; these contour lines at fixed levels show how the
    forest carves the space — regions reached by short random paths are the peaks.
    """
    try:
        grid = 60
        xs = np.linspace(x_range[0], x_range[1], grid)
        ys = np.linspace(y_range[0], y_range[1], grid)
        xx, yy = np.meshgrid(xs, ys)
        mesh = np.column_stack([xx.ravel(), yy.ravel()])
        _, zz = detector.predict(mesh)
        zz = zz.reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=zz,
                contours={
                    "start": 0.2,
                    "end": 0.8,
                    "size": 0.15,
                    "coloring": "lines",
                },
                line={"color": "rgba(255,255,255,0.5)", "width": 1},
                showscale=False,
                hoverinfo="skip",
            )
        )
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw IForest iso overlay")


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

    xx, yy, zz = score_mesh(detector, x_range, y_range)
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale="Inferno",
            showscale=False,
            opacity=0.6,
            contours={"showlines": False},
            hoverinfo="skip",
        )
    )

    if family in ("covariance_empirical", "covariance_robust", "covariance_elliptic"):
        _add_covariance_overlay(fig, detector, family)
    elif family == "knn":
        _add_knn_illustration(fig, X, scores, k_for_illustration)
    elif family == "lof":
        _add_lof_illustration(fig, X, k_for_illustration)
    elif family == "ocsvm":
        _add_ocsvm_boundary(fig, detector, x_range, y_range)
    elif family == "zscore":
        _add_zscore_band(fig, detector)
    elif family == "pca":
        _add_pca_axes(fig, detector, X)
    elif family == "iforest":
        _add_iforest_iso(fig, detector, x_range, y_range)

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
                    "colorscale": "Magenta",
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
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
            "range": x_range,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
            "range": y_range,
        },
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
        font={"color": TEXT, "size": 11},
        paper_bgcolor=BG_SURFACE,
        plot_bgcolor=BG_CANVAS,
        height=380,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
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


@st.cache_data(show_spinner=False, max_entries=256)
def _load_dataset(
    dataset_key: str, n_samples: int, contamination: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic dataset for the cache key; the same seed/args as the page."""
    X, y_true, _ = SyntheticDatasetGenerator.generate(
        dataset_key,
        n_samples=n_samples,
        contamination=contamination,
        random_state=42,
    )
    return X, y_true


@st.cache_data(show_spinner=False, max_entries=256)
def _compute_card(
    detector_name: str,
    param_items: tuple[tuple[str, Any], ...],
    dataset_key: str,
    n_samples: int,
    contamination: float,
) -> dict[str, Any]:
    """Fit, score and build the figure for one detector card.

    Cached keyed on the full parameter + dataset signature, so dragging a slider
    only recomputes this card while sibling cards reuse their cached figures.
    """
    X, y_true = _load_dataset(dataset_key, n_samples, contamination)
    params = dict(param_items)
    result = _fit_and_score(detector_name, params, X, y_true)
    fig = _build_figure(
        detector_name,
        result["detector"],
        X,
        result["y_pred"],
        result["scores"],
        int(params.get("n_neighbors", 5)),
    )
    return {"auroc": result["auroc"], "fig": fig}


def _render_detector_card(
    detector_name: str,
    dataset_key: str,
    n_samples: int,
    contamination: float,
) -> None:
    """Fit, chart and render one detector as a card (st.cache_data keyed by config)."""
    spec = DETECTOR_REGISTRY[detector_name]
    prefix = f"{detector_name}_{dataset_key}_{n_samples}"
    params = read_param_values(spec.params, prefix)
    # One global contamination control (top of the page) drives every detector's
    # assumed anomaly ratio, so the per-card sliders don't repeat across the grid.
    if "contamination" in params:
        params["contamination"] = contamination
    widget_params = tuple(p for p in spec.params if p.kwarg != "contamination")

    param_items = tuple(sorted(params.items()))
    result = _compute_card(
        detector_name, param_items, dataset_key, n_samples, contamination
    )
    # Mirror into session state so the ranking section can reuse the same AUROC.
    st.session_state[f"cache_{detector_name}"] = result

    detector_card(
        name=detector_name,
        category=spec.category,
        description=spec.description,
        fig=result["fig"],
        auroc=result["auroc"],
        params=widget_params,
        prefix=prefix,
        values=params,
        chart_key=f"chart_{prefix}",
    )


def _render_family_group(
    category: str,
    dataset_key: str,
    n_samples: int,
    contamination: float,
) -> None:
    """Render all detectors of one family with a colored header, two per row."""
    names = [n for n in TEACHING_DETECTORS if DETECTOR_REGISTRY[n].category == category]
    if not names:
        return

    family_header(category, category, CATEGORY_CAPTIONS.get(category, ""))

    for row_start in range(0, len(names), 2):
        cols = st.columns(2, gap="medium")
        for col_offset, col in enumerate(cols):
            idx = row_start + col_offset
            if idx >= len(names):
                break
            with col:
                _render_detector_card(names[idx], dataset_key, n_samples, contamination)


def render_playground_view() -> None:
    breadcrumb([("Data", True), ("Features", True), ("Detect", True)])
    st.markdown(
        """
        ## 2D Playground: How Anomaly Detectors Work

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
    categories: list[str] = []
    for name in TEACHING_DETECTORS:
        category = DETECTOR_REGISTRY[name].category
        if category not in categories:
            categories.append(category)
    for category in categories:
        _render_family_group(
            category, dataset_key_internal, n_samples, contamination_global
        )

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
    st.caption(
        "Didactic, not a verdict: a high AUROC here means the detector matches THIS "
        "geometry well. Real data is rarely this clean."
    )
    _, mid, _ = st.columns([0.5, 2, 0.5])
    with mid:
        st.dataframe(ranking_data, width="stretch", hide_index=True)

    st.markdown("---")
    with st.expander("Dataset Information"):
        metric_row(
            [
                ("Total Points", f"{len(X):,}"),
                ("Normal", f"{(y_true == 0).sum():,}"),
                ("Anomalies", f"{(y_true == 1).sum():,}"),
                ("Dimensions", "2D"),
            ]
        )
