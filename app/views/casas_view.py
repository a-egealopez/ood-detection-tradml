"""CASAS track: run the ensemble pipeline per house and inspect the results.

The view owns all configuration (main area, like the Features step) plus the result
tabs while the actual ML work lives in ``src.pipeline``. Runs automatically: any
configuration change recomputes the affected houses (cached by a config signature).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import (
    auroc_pill,
    breadcrumb,
    clickable_cards,
    colored_section_header,
    family_header,
    info_box,
    metric_row,
    read_param_values,
    render_param_widgets,
    render_resources,
)
from data_access import list_houses, load_house_events, load_house_events_injected
from detectors.factory import build_detector
from detectors.sequential.hawkes_detector import HawkesDetector
from mesh import GRID_RESOLUTION, contour_polylines, score_mesh
from streamlit_config import (
    DETECTOR_CATEGORIES,
    DETECTOR_DEFAULTS_LIST,
    DETECTOR_NAMES,
    DETECTOR_REGISTRY,
    ENSEMBLE_DEFAULTS,
)
from theme import (
    ANOMALY,
    ANOMALY_SCALE,
    MUTED,
    PRIMARY,
    PRIMARY_MID,
    PRIMARY_SOFT,
    SUCCESS,
    WARNING,
    apply_layout,
    display_chart,
    family_color,
    score_scale_css,
)

from features import FeatureScaler, TemporalFeatureExtractor


# ============================================================================
# MAIN-AREA CONFIGURATION (moved out of the sidebar, like the Features step)
# ============================================================================
def _detector_params_from_session(source: str, names: list[str]) -> dict:
    """Reassemble ``{name: {kwarg: value}}`` from the tuning sliders of each card."""
    params = {}
    for name in names:
        spec = DETECTOR_REGISTRY.get(name)
        if spec is None:
            continue
        params[name] = read_param_values(spec.params, f"adv_{source}_{name}")
    return params


def _render_ensemble_config(source: str) -> dict:
    """Ensemble rule: pick how the per-detector scores combine into one verdict.

    Two clickable cards (Weighted / Majority) matching the picker style of the
    other steps. The advanced-customize toggle and the synthetic-evaluation
    panel were dropped (AUROC benchmarking lives in the 2D/synthetic track).
    """
    card_specs = [
        {
            "id": "weighted",
            "icon": "⚖️",
            "title": "Weighted (soft)",
            "description": "Blends each detector's continuous score into a "
            "weighted average — uses all the information.",
            "badge": "continuous",
            "color": PRIMARY,
        },
        {
            "id": "majority",
            "icon": "🗳️",
            "title": "Majority (hard)",
            "description": "Each detector votes yes/no; a day is flagged only "
            "when most detectors agree.",
            "badge": "votes",
            "color": SUCCESS,
        },
    ]
    ensemble_id = clickable_cards(card_specs, key=f"ens_rule_{source}")
    ensemble_mode = "soft" if ensemble_id == "weighted" else "hard"

    return {
        "ensemble_mode": ensemble_mode,
        "threshold_percentile": float(ENSEMBLE_DEFAULTS["threshold_percentile"]),
    }


def _resolved_houses(source: str) -> list[str]:
    """All houses available for the source - houses are never picked per-step."""
    try:
        return list(list_houses(source))
    except Exception:  # noqa: BLE001 - DB may not exist until ingestion runs
        return []


def _toggle_detector(store_key: str, name: str) -> None:
    """Flip a detector's membership in the ensemble selection."""
    current = set(st.session_state.get(store_key, DETECTOR_DEFAULTS_LIST))
    current.symmetric_difference_update([name])
    st.session_state[store_key] = [n for n in DETECTOR_NAMES if n in current]


@dataclass
class HouseFeatures:
    """Scaled daily features of one house + the didactic PCA(2) projection.

    ``train_n`` is the number of training rows (the 70% split) used to fit the
    scaler and the PCA - only training rows feed the fits, never the test tail.
    ``X_counts`` is the raw (unscaled) daily event-count subset, used only by
    detectors that need genuine counts (Hawkes).
    """

    X_scaled: np.ndarray
    X_2d: np.ndarray
    pca: Any
    dates: list[str]
    train_n: int
    X_counts: np.ndarray


def _house_features_cached(
    source: str, house_id: str, scenario: str = "control", intensity: str = "medium"
) -> "HouseFeatures":
    """Scaled daily features for one house (cached).

    ``X_2d`` is a PCA(2) projection of the scaled features, fitted on the training
    rows only - a didactic 2-D view for the score-cloud charts (the mesh gradient
    maps back through ``pca``), not the true decision boundary of any detector.
    """
    from sklearn.decomposition import PCA

    from pipeline import MIN_DAYS

    if source == "synthetic" and scenario != "control":
        df, _ = load_house_events_injected(source, house_id, scenario, intensity)
    else:
        df = load_house_events(source, house_id)
    if df is None or df.empty:
        return HouseFeatures((), (), None, [], 0, ())
    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df)
    if len(X) < MIN_DAYS:
        return HouseFeatures((), (), None, [], 0, ())
    X_counts = extractor.count_columns(X)
    X_scaled = FeatureScaler().fit_transform(X)
    train_n = max(1, int(len(X_scaled) * 0.7))
    pca = PCA(n_components=2).fit(X_scaled[:train_n])
    X_2d = pca.transform(X_scaled)
    return HouseFeatures(X_scaled, X_2d, pca, list(dates), train_n, X_counts)


def _injection_config(source: str) -> tuple[str, str]:
    """Anomaly scenario chosen in the Data step (only applies to synthetic)."""
    if source == "synthetic":
        scenario = st.session_state.get("fx_scenario", "control")
        intensity = st.session_state.get("fx_scenario_intensity", "medium")
        return scenario, intensity
    return "control", "medium"


def _render_detector_card(
    source: str,
    preview_house: str,
    name: str,
    selected: bool,
    features: HouseFeatures,
) -> None:
    """One detector card: toggle header + chart (timeline / PCA cloud) + sliders."""
    spec = DETECTOR_REGISTRY[name]
    st.button(
        f"{'✅' if selected else '▫️'} {name}",
        key=f"toggle_{source}_{name}",
        use_container_width=True,
        type="primary" if selected else "secondary",
        on_click=_toggle_detector,
        args=(f"det_names_{source}", name),
        help="Click to include / exclude this detector from the ensemble.",
    )

    values = _detector_params_from_session(source, [name])[name]
    try:
        detector = build_detector(name, values)
        uses_counts = isinstance(detector, HawkesDetector)
        X_fit = features.X_counts if uses_counts else features.X_scaled
        detector.fit(X_fit[: features.train_n])
        y_pred, scores = detector.predict(X_fit)
        if uses_counts:
            # The PCA-plane score map is fitted on the scaled features; a
            # count-based intensity model cannot be drawn over it honestly, so
            # the Hawkes card only shows the timeline.
            st.caption(
                "Applied to the raw daily event counts (n_events, n_sensors, "
                "activity_hours) - a Poisson intensity process, not the scaled "
                "feature matrix. The score map is only meaningful for "
                "vectorial detectors."
            )
            fig = _build_detector_timeline(features.dates, scores, name)
        else:
            view = st.segmented_control(
                "Visualization",
                ["Time", "Score map"],
                default="Score map",
                label_visibility="collapsed",
                key=f"card_view_{source}_{preview_house}_{name}",
            )
            if view == "Score map":
                fig = _build_detector_cloud(
                    name, features.X_2d, features.pca, scores, detector, y_pred, features.dates
                )
            else:
                fig = _build_detector_timeline(features.dates, scores, name)
        display_chart(fig, key=f"det_chart_{source}_{preview_house}_{name}")
    except Exception as exc:  # noqa: BLE001 - fragile deps (e.g. tick) may raise
        st.warning(f"⚠️ {name}: {exc}")

    render_param_widgets(spec.params, f"adv_{source}_{name}", values, help=True)


def _render_detect_panel(source: str) -> None:
    """Detect step: one tuneable, clickable card per detector, grouped by family."""
    houses = _resolved_houses(source)
    if not houses:
        st.warning(
            "No houses loaded. Run first `python src/ingestion/casas_loader.py` or "
            "pick houses in the Data step (Real data)."
        )
        st.stop()

    preview_house = st.selectbox(
        "House to visualize", houses, key=f"preview_house_{source}"
    )
    scenario, intensity = _injection_config(source)
    features = _house_features_cached(source, preview_house, scenario, intensity)
    if not len(features.dates):
        st.warning("Not enough days in this house to fit the detectors.")
        st.stop()

    if scenario != "control":
        st.caption(
            f"Anomaly scenario **{scenario}** ({intensity}) is applied to this "
            "house — the scores below reflect the injected stream."
        )

    if f"det_names_{source}" not in st.session_state:
        st.session_state[f"det_names_{source}"] = list(DETECTOR_DEFAULTS_LIST)
    selected = set(st.session_state[f"det_names_{source}"])

    st.caption(
        "Score map: the line with the dark edge is each detector's **decision "
        "boundary**; background shades the anomaly score of the PCA plane "
        "(one dot = one day)."
    )
    st.markdown(
        "<div class='score-legend'>"
        "<span><b>Background</b> = anomaly score</span>"
        f"<span class='score-bar' style='background:{score_scale_css()};'></span>"
        "<span class='low-high'><span>normal</span><span>→</span><span>anomalous</span></span>"
        "</div>",
        unsafe_allow_html=True,
    )

    for category, names in DETECTOR_CATEGORIES.items():
        family_header(category, category)
        for row_start in range(0, len(names), 2):
            cols = st.columns(2, gap="medium")
            for offset, col in enumerate(cols):
                idx = row_start + offset
                if idx >= len(names):
                    break
                with col:
                    _render_detector_card(
                        source,
                        preview_house,
                        names[idx],
                        names[idx] in selected,
                        features,
                    )

    st.markdown("---")
    n_selected = len(selected)
    st.caption(
        f"**{n_selected} of {len(DETECTOR_NAMES)}** detectors selected — those will "
        "feed the ensemble in the next step."
    )


def _resolve_detect_config(source: str) -> dict:
    """Detect-side config read back for the Ensemble step (no widgets rendered).

    Houses are all the available ones for the source; detectors are
    the ones selected on the Detect cards by default DETECTOR_DEFAULTS_LIST.
    """
    houses = _resolved_houses(source)
    if not houses:
        st.warning(
            f"No data loaded for {source}. Run first "
            "`python src/ingestion/casas_loader.py`."
        )
        st.stop()

    selected = st.session_state.get(f"det_names_{source}", DETECTOR_DEFAULTS_LIST)
    detector_names = [n for n in DETECTOR_NAMES if n in selected]

    return {
        "houses": houses,
        "detector_names": detector_names,
        "detector_params": _detector_params_from_session(source, detector_names),
    }


def _combine_config(source: str, detect: dict, ensemble: dict) -> dict:
    """Merge the Detect-step and Ensemble-step pieces into one pipeline config."""
    scenario, intensity = _injection_config(source)
    pipeline_kwargs = {
        "detector_names": detect["detector_names"],
        "detector_params": detect["detector_params"],
        "ensemble_mode": ensemble["ensemble_mode"],
        "threshold_percentile": ensemble["threshold_percentile"],
    }
    signatures = (
        source,
        tuple(sorted(detect["houses"])),
        tuple(detect["detector_names"]),
        tuple(
            sorted(
                (d, tuple(sorted(p.items())))
                for d, p in detect["detector_params"].items()
            )
        ),
        ensemble["ensemble_mode"],
        ensemble["threshold_percentile"],
        scenario,
        intensity,
    )
    return {
        **detect,
        **ensemble,
        "scenario": scenario,
        "intensity": intensity,
        "pipeline_kwargs": pipeline_kwargs,
        "signature": signatures,
    }


# ============================================================================
# AUTO-RUN (cached by config signature, no button)
# ============================================================================
def _run_pipeline(source: str, config: dict) -> dict:
    from pipeline import run_house

    results = st.session_state.get(f"results_{source}", {})
    if st.session_state.get(f"sig_{source}") == config["signature"]:
        return results

    if not config["houses"]:
        st.warning("Select at least one house in the Data step.")
        return results

    new_results = {}
    new_anom = {}
    for house_id in config["houses"]:
        scenario = config.get("scenario", "control")
        if source == "synthetic" and scenario != "control":
            df, anomaly_dates = load_house_events_injected(
                source, house_id, scenario, config.get("intensity", "medium")
            )
        else:
            df = load_house_events(source, house_id)
            anomaly_dates = ()
        try:
            result, error = run_house(house_id, df, **config["pipeline_kwargs"])
        except Exception as e:  # noqa: BLE001 - fragile deps (e.g. tick) may raise
            result, error = None, str(e)
        if error:
            st.warning(f"[{house_id}] {error}")
        else:
            new_results[house_id] = result
            new_anom[house_id] = tuple(anomaly_dates)

    st.session_state[f"sig_{source}"] = config["signature"]
    st.session_state[f"results_{source}"] = new_results
    st.session_state[f"anom_{source}"] = new_anom
    return new_results


# ============================================================================
# RESULT TABS
# ============================================================================
def _render_injected_eval(
    source: str,
    house_view: str,
    details: pd.DataFrame,
    anomaly_dates: tuple,
    score_cols: list[str],
) -> None:
    """AUROC of each detector against the injected days (holdout tail only).

    The synthetic scenario gives us ground-truth labels for the first time in the
    CASAS track, so the ensemble results can answer "did THIS detector catch THIS
    anomaly type?" — the same question the CLI matrix answers per cell.
    """
    colored_section_header("🎯", "Injected anomaly evaluation", WARNING)
    scenario = st.session_state.get("fx_scenario", "control")
    intensity = st.session_state.get("fx_scenario_intensity", "medium")
    st.caption(
        f"Scenario **{scenario}** ({intensity}) — "
        f"{len(anomaly_dates)} injected day(s): "
        + ", ".join(str(d) for d in anomaly_dates)
        + " (AUROC measured on the holdout tail only)."
    )

    dates = list(details["date"])
    split_idx = max(1, int(len(dates) * 0.7))
    holdout = np.zeros(len(dates), dtype=bool)
    holdout[split_idx:] = True
    y = np.asarray([d in set(anomaly_dates) for d in dates], dtype=int)

    labels = score_cols + ["ensemble_score"]
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels, strict=True):
        with col:
            try:
                auroc = float(roc_auc_score(y[holdout], details[label].values[holdout]))
            except ValueError:
                auroc = None
            auroc_pill(auroc, label=label.replace("_score", ""))

    st.caption(
        "Expected winners: **point** → the distance / vectorial family (Z-Score, "
        "Mahalanobis, IForest, PCA Reconstruction); **contextual** → Z-Score / HMM / "
        "Hawkes (the order models stay blind, ~0.5); **collective** → the Markov "
        "next-event model (the distance / context family stays blind)."
    )


def _render_house_results(
    source: str, house_view: str, r: Any, anomaly_dates: tuple = ()
) -> None:
    """Full result section for one house (houses are chosen in the Data step)."""
    details, anomalies, scores = r.details, r.anomalies, r.scores
    threshold = getattr(r.ensemble, "threshold", None)

    colored_section_header("📊", f"Anomaly Detection - {house_view}", ANOMALY)
    metric_row(
        [
            ("Total Anomalies", str(int(anomalies.sum()))),
            ("Anomaly Rate", f"{anomalies.mean() * 100:.2f}%"),
            ("Avg Score", f"{scores.mean():.3f}"),
        ]
    )

    score_cols = [
        c for c in details.columns if c.endswith("_score") and c != "ensemble_score"
    ]
    customdata = list(zip(*[details[c].round(3) for c in score_cols], strict=True))

    fig_timeline = go.Figure(
        go.Scatter(
            x=details["date"],
            y=details["ensemble_score"],
            mode="markers",
            marker={
                "color": anomalies,
                "colorscale": [[0, PRIMARY_SOFT], [1, ANOMALY]],
                "size": 9,
                "showscale": False,
                "line": {"width": 0.5, "color": "rgba(255,255,255,0.35)"},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>ensemble: %{y:.3f}"
                + "".join(
                    f"<br>{c}: %{{customdata[{i}]}}" for i, c in enumerate(score_cols)
                )
                + "<extra></extra>"
            ),
            name="Ensemble Score",
        )
    )
    if anomaly_dates:
        # Injected days (ground truth) shaded behind the scores: the anomaly block
        # sits on the holdout tail, so detection means raising the score there.
        fig_timeline.add_vrect(
            x0=pd.Timestamp(min(anomaly_dates)) - pd.Timedelta(hours=12),
            x1=pd.Timestamp(max(anomaly_dates)) + pd.Timedelta(hours=12),
            fillcolor=WARNING,
            opacity=0.14,
            line_width=0,
            annotation_text="injected",
            annotation_position="top left",
        )
    if threshold is not None:
        mode = r.ensemble.ensemble_mode
        label = (
            f"majority ({threshold:.2f})"
            if mode == "hard"
            else f"threshold (p{getattr(r.ensemble, 'ensemble_threshold_percentile', '?')})"
        )
        fig_timeline.add_hline(
            y=threshold,
            line={"color": ANOMALY, "width": 1.5, "dash": "dash"},
            annotation_text=label,
            annotation_position="top right",
        )
    apply_layout(
        fig_timeline, "Timeline: ensemble scores (red = above threshold)", height=460
    )
    display_chart(fig_timeline, key=f"timeline_{source}_{house_view}")
    st.caption("Red = above the threshold percentile · each point = one day.")

    if anomaly_dates:
        _render_injected_eval(source, house_view, details, anomaly_dates, score_cols)

    fig_hist = go.Figure(
        go.Histogram(
            x=scores, nbinsx=30, marker_color=PRIMARY_SOFT, name="Score Distribution"
        )
    )
    if threshold is not None:
        fig_hist.add_vline(
            x=threshold,
            line={"color": ANOMALY, "width": 1.5, "dash": "dash"},
            annotation_text="threshold",
        )
    apply_layout(fig_hist, "Score Distribution", height=320)
    display_chart(fig_hist, key=f"hist_{source}_{house_view}")

    colored_section_header("🚩", "Days flagged as anomalous", ANOMALY)
    anomalous_days = details[details["is_anomaly"] == 1].sort_values(
        "ensemble_score", ascending=False
    )
    st.dataframe(anomalous_days, width="stretch")

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

    csv_data = details.to_csv(index=False)
    st.download_button(
        "Download results (CSV)",
        data=csv_data,
        file_name=f"anomaly_predictions_{house_view}.csv",
        mime="text/csv",
    )


def _render_anomalies_tab(results: dict, source: str) -> None:
    """Ensemble results for one house at a time (all houses always computed)."""
    if not results:
        st.info(
            "Select at least one house in the Data step; results update automatically."
        )
        return

    house_view = st.selectbox(
        "House to display",
        list(results.keys()),
        key=f"anom_house_{source}",
        help="All houses run the pipeline; this picker only chooses which one to inspect.",
    )
    anomaly_dates = st.session_state.get(f"anom_{source}", {}).get(house_view, ())
    _render_house_results(source, house_view, results[house_view], anomaly_dates)

    info_box(
        "⚡",
        "Quick tip",
        "Compare the per-detector decision boundaries on the 2D Playground (teaching "
        "track) for intuition on what each family does.",
        color=MUTED,
    )


def _build_detector_timeline(dates: list, scores: np.ndarray, name: str) -> go.Figure:
    """Compact score-over-time chart for one detector (used per detect card)."""
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=scores,
            mode="markers+lines",
            marker={"size": 6, "color": PRIMARY_SOFT},
            line={"width": 1.2, "color": PRIMARY_SOFT},
            name=name,
            hovertemplate="%{x}<br>score: %{y:.3f}<extra></extra>",
        )
    )
    apply_layout(fig, f"{name} — anomaly score over time", height=250)
    fig.update_yaxes(title_text="score", range=[0, 1])
    fig.update_xaxes(title_text="Date")
    return fig


def _build_detector_cloud(
    name: str,
    X_2d: np.ndarray,
    pca: Any,
    scores: np.ndarray,
    detector: Any,
    y_pred: np.ndarray,
    dates: list,
) -> go.Figure:
    """2-D score map over the PCA plane, styled like the 2D Playground.

    The detector's real score field is drawn over the grid with the shared
    anomaly-score colorscale, its real decision boundary is traced with marching
    squares (same three-layer seam as the playground), and each daily point is
    marked as normal (indigo) or anomalous (pink halo) using the detector's own
    verdict.
    """
    spec = DETECTOR_REGISTRY[name]
    boundary_color = family_color(spec.category)

    x_margin = 0.1 * (X_2d[:, 0].max() - X_2d[:, 0].min() + 1e-6)
    y_margin = 0.1 * (X_2d[:, 1].max() - X_2d[:, 1].min() + 1e-6)
    x_range = (X_2d[:, 0].min() - x_margin, X_2d[:, 0].max() + x_margin)
    y_range = (X_2d[:, 1].min() - y_margin, X_2d[:, 1].max() + y_margin)

    _, _, zz, threshold = score_mesh(
        detector, x_range, y_range, transform=pca.inverse_transform
    )
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], GRID_RESOLUTION),
            y=np.linspace(y_range[0], y_range[1], GRID_RESOLUTION),
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
    # Decision boundary: the iso-line at the detector's split threshold, drawn as
    # plain Scatter lines (guaranteed to render) with the family-colored seam.
    polylines = contour_polylines(zz, threshold, x_range, y_range)
    if polylines:
        for line_color, width in [
            ("rgba(8, 12, 20, 0.85)", 5.0),
            (boundary_color, 3.0),
            ("rgba(255, 255, 255, 0.95)", 1.2),
        ]:
            bx: list[float | None] = []
            by: list[float | None] = []
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

    normal_mask = y_pred == 0
    fig.add_trace(
        go.Scatter(
            x=X_2d[normal_mask, 0],
            y=X_2d[normal_mask, 1],
            mode="markers",
            marker={
                "size": 8,
                "color": PRIMARY_MID,
                "line": {"width": 0.5, "color": "rgba(255,255,255,0.5)"},
            },
            customdata=[[dates[i]] for i in np.flatnonzero(normal_mask)],
            hovertemplate="%{customdata[0]}<br>score: %{text:.3f}<extra></extra>",
            text=scores[normal_mask],
            showlegend=False,
        )
    )
    anomaly_mask = y_pred == 1
    if anomaly_mask.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=X_2d[anomaly_mask, 0],
                y=X_2d[anomaly_mask, 1],
                mode="markers",
                marker={
                    "size": 9,
                    "color": ANOMALY,
                    "line": {"width": 1.2, "color": "rgba(255,255,255,0.9)"},
                },
                customdata=[[dates[i]] for i in np.flatnonzero(anomaly_mask)],
                hovertemplate="%{customdata[0]}<br>score: %{text:.3f}<extra></extra>",
                text=scores[anomaly_mask],
                showlegend=False,
            )
        )
    apply_layout(fig, name, height=320)
    fig.update_xaxes(title_text="PC1", range=x_range)
    fig.update_yaxes(title_text="PC2", range=y_range)
    return fig


# ============================================================================
# VIEW ENTRY POINT
# ============================================================================
def render_casas_view(source: str, stage: str = "detect") -> None:
    """CASAS track. ``stage`` picks the guided-workflow focus:

    - ``"detect"`` -> one tuneable, clickable card per detector, grouped by family.
    - ``"ensemble"`` -> how the selected detectors combine into one anomaly verdict.
    """
    focus = "ensemble" if stage == "ensemble" else "detect"
    breadcrumb(
        [
            ("Data", False),
            ("Features", False),
            ("Detect", focus == "detect"),
            ("Ensemble", focus == "ensemble"),
        ]
    )

    if focus == "detect":
        colored_section_header(
            "🧠", f"Detect anomalies - {source.title()} data", PRIMARY
        )
        st.caption(
            "Tune and enable the detectors you trust below, grouped by family. "
            "The selected ones feed the ensemble in the next step."
        )
        _render_detect_panel(source)
        return

    detect = _resolve_detect_config(source)
    ensemble = _render_ensemble_config(source)
    render_resources("Ensemble detectors")

    config = _combine_config(source, detect, ensemble)
    results = _run_pipeline(source, config)

    _render_anomalies_tab(results, source)
