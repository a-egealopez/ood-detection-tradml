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
    read_param_values,
    section_title,
)
from mesh import contour_polylines, score_mesh  # noqa: E402
from streamlit_config import DETECTOR_REGISTRY  # noqa: E402
from theme import (  # noqa: E402
    ANOMALY,
    ANOMALY_SCALE,
    ANOMALY_SOFT,
    BG_CANVAS,
    BG_SURFACE,
    BORDER,
    PRIMARY,
    PRIMARY_MID,
    PRIMARY_SOFT,
    TEXT,
    family_color,
    score_scale_css,
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

# Covariance-ellipse levels for the Gaussian-family overlays: the concentric
# rings ARE the covariance of the fitted Gaussian (variance + correlation),
# one ring per standard deviation. Drawn rings and the corner legend share this
# single definition so they can never drift apart.
# Tuple: (n_std, trace_color, legend_color, label).
ELLIPSE_LEVELS = [
    (1, PRIMARY_SOFT, PRIMARY, "1σ"),
    (2, "rgba(100,116,139,0.7)", "#94a3b8", "2σ"),
    (3, ANOMALY_SOFT, ANOMALY, "3σ"),
]


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
    for n_std, color, text_color, label in ELLIPSE_LEVELS:
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
        rightmost = int(np.argmax(ex))
        fig.add_annotation(
            x=ex[rightmost],
            y=ey[rightmost],
            text=label,
            showarrow=False,
            font={"size": 10, "color": text_color, "weight": "bold"},
            xshift=10,
            bgcolor="rgba(248, 250, 252, 0.85)",
            bordercolor=text_color,
            borderwidth=1,
            borderpad=3,
        )


def _add_covariance_overlay(fig: go.Figure, detector: Any, family: str) -> None:
    """Draw the 1/2/3-sigma ellipses of the covariance the detector fitted.

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
    """Lines to the k neighbors of the top anomaly, weighted by distance."""
    target_idx = int(np.argmax(scores))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    dists, neighbor_idx = nn.kneighbors(X[target_idx].reshape(1, -1))
    neighbor_ids = neighbor_idx[0][1:]
    neighbor_dists = dists[0][1:]
    max_d = float(neighbor_dists.max()) if neighbor_dists.size else 1.0

    for idx, dist in zip(neighbor_ids, neighbor_dists, strict=True):
        weight = 1.0 - 0.6 * (dist / max_d if max_d > 0 else 0.0)
        fig.add_trace(
            go.Scatter(
                x=[X[target_idx, 0], X[idx, 0]],
                y=[X[target_idx, 1], X[idx, 1]],
                mode="lines",
                line={"color": ANOMALY_SOFT, "width": 1 + weight * 2},
                opacity=float(weight),
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
    fig.add_annotation(
        x=X[target_idx, 0],
        y=X[target_idx, 1],
        text=f"top anomaly · {k} nearest neighbors",
        showarrow=True,
        arrowhead=0,
        ax=0,
        ay=-24,
        font={"size": 9, "color": "rgba(255,255,255,0.7)"},
        bgcolor="rgba(0,0,0,0.3)",
        borderpad=3,
    )


def _add_lof_illustration(
    fig: go.Figure, X: np.ndarray, k: int, sample_size: int | None = None
) -> None:
    """Circles of radius = distance to the k-th neighbor illustrate local density."""
    if sample_size is None:
        sample_size = max(8, min(20, len(X) // 15))
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
                line={"color": PRIMARY_SOFT, "width": 1.2},
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    # Single didactic label (same light-background style as the covariance rings):
    # anchored on the top of the largest circle, in a sparse region where it does
    # not collide with data.
    if sample_idx.size:
        biggest = sample_idx[int(np.argmax(distances[sample_idx, -1]))]
        radius = distances[biggest, -1]
        fig.add_annotation(
            x=X[biggest, 0],
            y=X[biggest, 1] + radius,
            yanchor="bottom",
            text="radius = distance to<br>the k-th neighbor",
            showarrow=False,
            align="center",
            font={"size": 9, "color": "#0f172a", "weight": "bold"},
            bgcolor="rgba(248, 250, 252, 0.85)",
            bordercolor=PRIMARY,
            borderwidth=1,
            borderpad=4,
        )


def _add_zscore_band(fig: go.Figure, detector: Any) -> None:
    """Draw the axis-aligned threshold rectangle mu +/- threshold*sigma per feature.

    Z-Score is univariate: each feature is judged independently, so its "normal"
    region is an axis-parallel box, not a rotated ellipse. This rectangle makes
    that explicit.
    """
    try:
        mu, sigma, threshold = detector.mu, detector.sigma, detector.threshold
        x0, x1 = mu[0] - threshold * sigma[0], mu[0] + threshold * sigma[0]
        y0, y1 = mu[1] - threshold * sigma[1], mu[1] + threshold * sigma[1]
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
        fig.add_annotation(
            x=x0,
            y=y1,
            text="±k·σ per axis",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 9, "color": "white"},
            bgcolor="rgba(0,0,0,0.3)",
            borderpad=3,
        )
    except Exception:  # noqa: BLE001 - overlay is cosmetic; never break the chart
        logger.warning("Could not draw Z-Score band overlay")


def _build_figure(
    detector_name: str,
    detector: Any,
    X: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    k_for_illustration: int,
) -> go.Figure:
    spec = DETECTOR_REGISTRY[detector_name]
    family = spec.family
    boundary_color = family_color(spec.category)

    x_margin = 0.1 * (X[:, 0].max() - X[:, 0].min() + 1e-6)
    y_margin = 0.1 * (X[:, 1].max() - X[:, 1].min() + 1e-6)
    x_range = (X[:, 0].min() - x_margin, X[:, 0].max() + x_margin)
    y_range = (X[:, 1].min() - y_margin, X[:, 1].max() + y_margin)

    fig = go.Figure()

    xx, yy, zz, threshold = score_mesh(detector, x_range, y_range)

    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            zmin=0,
            zmax=1,
            colorscale=ANOMALY_SCALE,
            showscale=False,
            opacity=0.6,
            contours={"coloring": "heatmap", "showlines": False},
            hoverinfo="skip",
        )
    )
    # Decision boundary: the iso-line of the score field at the detector's split
    # threshold. Extracted with marching squares and drawn as plain Scatter lines
    # (a single-level Plotly contour isoline can silently not render), as a crisp
    # three-layer "seam": dark outline separates it from bright zones, the family
    # accent carries the identity, a white core keeps it visible on dark zones.
    polylines = contour_polylines(zz, threshold, x_range, y_range)
    if polylines:
        for line_color, width in [
            ("rgba(8, 12, 20, 0.85)", 5.0),
            (boundary_color, 3.0),
            ("rgba(255, 255, 255, 0.95)", 1.2),
        ]:
            bx: list[float] = []
            by: list[float] = []
            for pl in polylines:
                bx.extend(pl[:, 0].tolist())
                by.extend(pl[:, 1].tolist())
                bx.append(None)
                by.append(None)
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line={"color": line_color, "width": width},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    if family in ("covariance_empirical", "covariance_robust", "covariance_elliptic"):
        _add_covariance_overlay(fig, detector, family)
    elif family == "knn":
        _add_knn_illustration(fig, X, scores, k_for_illustration)
    elif family == "lof":
        _add_lof_illustration(fig, X, k_for_illustration)
    elif family == "zscore":
        _add_zscore_band(fig, detector)

    normal_mask = y_pred == 0
    fig.add_trace(
        go.Scatter(
            x=X[normal_mask, 0],
            y=X[normal_mask, 1],
            mode="markers",
            marker={
                "size": 5,
                "color": PRIMARY_MID,
                "line": {"width": 1, "color": "rgba(255,255,255,0.5)"},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    anomaly_mask = y_pred == 1
    if anomaly_mask.sum() > 0:
        # A single strong accent (not score-colored: the background field already
        # encodes the score) with a white halo so anomalies pop on any tint.
        fig.add_trace(
            go.Scatter(
                x=X[anomaly_mask, 0],
                y=X[anomaly_mask, 1],
                mode="markers",
                marker={
                    "size": 8,
                    "color": ANOMALY,
                    "line": {"width": 1.2, "color": "rgba(255,255,255,0.9)"},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "showline": True,
            "linecolor": BORDER,
            "range": x_range,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "showline": True,
            "linecolor": BORDER,
            "range": y_range,
        },
        hovermode=False,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    fig.update_layout(
        title={
            "text": f"<b>{detector_name}</b>",
            "font": {"size": 16, "color": TEXT},
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
    X, y_true = SyntheticDatasetGenerator.generate(
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
    st.markdown("## 2D Playground: How Anomaly Detectors Work")

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

    st.markdown("---")

    section_title("All Detectors at a Glance")
    st.caption(
        "The line with the dark edge is the **decision boundary** — the detector's "
        "split between normal and anomalous regions. The background shades the "
        "anomaly score of every region."
    )
    st.markdown(
        "<div class='score-legend'>"
        "<span><b>Background</b> = anomaly score</span>"
        f"<span class='score-bar' style='background:{score_scale_css()};'></span>"
        "<span class='low-high'><span>normal</span><span>→</span><span>anomalous</span></span>"
        "</div>",
        unsafe_allow_html=True,
    )
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
