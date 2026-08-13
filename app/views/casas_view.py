"""CASAS track: run the ensemble pipeline per house and inspect the results.

The view owns all configuration (main area, like the Features step) plus the result
tabs while the actual ML work lives in ``src.pipeline``. Runs automatically: any
configuration change recomputes the affected houses (cached by a config signature).
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import (
    breadcrumb,
    clickable_cards,
    colored_section_header,
    family_header,
    info_box,
    metric_row,
)
from data_access import list_houses, load_house_events
from streamlit_config import (
    DETECTOR_CATEGORIES,
    DETECTOR_DEFAULTS_LIST,
    DETECTOR_NAMES,
    DETECTOR_REGISTRY,
    ENSEMBLE_DEFAULTS,
    SYNTHETIC_EVALUATION_DEFAULTS,
)
from theme import (
    ANOMALY,
    MUTED,
    PRIMARY,
    PRIMARY_SOFT,
    SUCCESS,
    apply_layout,
    display_chart,
    family_color,
)

from features import FeatureScaler, TemporalFeatureExtractor


# ============================================================================
# MAIN-AREA CONFIGURATION (moved out of the sidebar, like the Features step)
# ============================================================================
def _detector_param_values(source: str, name: str) -> dict:
    """Current tuning values for a detector (read from the sliders' session keys)."""
    spec = DETECTOR_REGISTRY[name]
    return {
        p.kwarg: st.session_state.get(f"adv_{source}_{name}_{p.kwarg}", p.default)
        for p in spec.params
    }


def _detector_params_from_session(source: str, names: list[str]) -> dict:
    """Reassemble ``{name: {kwarg: value}}`` from the tuning sliders of each card."""
    params = {}
    for name in names:
        spec = DETECTOR_REGISTRY.get(name)
        if spec is None:
            continue
        params[name] = {
            p.kwarg: st.session_state.get(f"adv_{source}_{name}_{p.kwarg}", p.default)
            for p in spec.params
        }
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
        "weighting_scheme": "Uniform",
        "threshold_percentile": float(ENSEMBLE_DEFAULTS["threshold_percentile"]),
        "contamination": SYNTHETIC_EVALUATION_DEFAULTS["contamination"],
        "magnitude": SYNTHETIC_EVALUATION_DEFAULTS["magnitude"],
    }


def _resolved_houses(source: str) -> list[str]:
    """All houses available for the source — houses are never picked per-step."""
    try:
        return list(list_houses(source))
    except Exception:
        return []


def _toggle_detector(store_key: str, name: str) -> None:
    """Flip a detector's membership in the ensemble selection."""
    current = set(st.session_state.get(store_key, DETECTOR_DEFAULTS_LIST))
    current.symmetric_difference_update([name])
    st.session_state[store_key] = [n for n in DETECTOR_NAMES if n in current]


@st.cache_data(show_spinner=False)
def _house_features_cached(source: str, house_id: str) -> tuple:
    """Scaled daily features for one house (cached).

    Returns ``(X, X_2d, pca, dates, train_n)``. ``X_2d`` is a PCA(2) projection of
    the scaled features, fitted on the training rows only — a didactic 2-D view
    for the score-cloud charts (the mesh gradient maps back through ``pca``), not
    the true decision boundary of any detector.
    """
    from sklearn.decomposition import PCA

    from pipeline import MIN_DAYS

    df = load_house_events(source, house_id)
    if df is None or df.empty:
        return (), (), None, [], 0
    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df)
    if len(X) < MIN_DAYS:
        return (), (), None, [], 0
    X_scaled = FeatureScaler().fit_transform(X)
    train_n = max(1, int(len(X_scaled) * 0.7))
    pca = PCA(n_components=2).fit(X_scaled[:train_n])
    X_2d = pca.transform(X_scaled)
    return X_scaled, X_2d, pca, list(dates), train_n


def _render_detector_card(
    source: str,
    preview_house: str,
    name: str,
    selected: bool,
    X_scaled: np.ndarray,
    X_2d: np.ndarray,
    pca: Any,
    dates: list,
    train_n: int,
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

    values = _detector_param_values(source, name)
    try:
        detector = spec.detector_cls(**values)
        detector.fit(X_scaled[:train_n])
        _, scores = detector.predict(X_scaled)
        view = st.segmented_control(
            "Visualization",
            ["Time", "Score cloud"],
            default="Score cloud",
            label_visibility="collapsed",
            key=f"card_view_{source}_{preview_house}_{name}",
        )
        if view == "Score cloud":
            fig = _build_detector_cloud(name, X_2d, pca, scores, detector)
        else:
            fig = _build_detector_timeline(dates, scores, name)
        display_chart(fig, key=f"det_chart_{source}_{preview_house}_{name}")
    except Exception as exc:  # noqa: BLE001 - fragile deps (e.g. tick) may raise
        st.warning(f"⚠️ {name}: {exc}")

    for p in spec.params:
        key = f"adv_{source}_{name}_{p.kwarg}"
        if p.options:
            st.selectbox(
                p.label,
                list(p.options),
                index=list(p.options).index(st.session_state.get(key, p.default)),
                help=f"Default: {p.default}",
                key=key,
            )
        else:
            ptype = type(p.default)
            st.slider(
                p.label,
                ptype(p.min),
                ptype(p.max),
                ptype(st.session_state.get(key, p.default)),
                step=ptype(p.step),
                key=key,
            )


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
    X_scaled, X_2d, pca, dates, train_n = _house_features_cached(source, preview_house)
    if not len(dates):
        st.warning("Not enough days in this house to fit the detectors.")
        st.stop()

    if f"det_names_{source}" not in st.session_state:
        st.session_state[f"det_names_{source}"] = list(DETECTOR_DEFAULTS_LIST)
    selected = set(st.session_state[f"det_names_{source}"])

    for category, names in DETECTOR_CATEGORIES.items():
        family_header(category, family_color(category))
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
                        X_scaled,
                        X_2d,
                        pca,
                        dates,
                        train_n,
                    )

    st.markdown("---")
    n_sel = len(selected)
    st.caption(
        f"**{n_sel} of {len(DETECTOR_NAMES)}** detectors selected — those will "
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
        "advanced": True,
        "detector_names": detector_names,
        "detector_params": _detector_params_from_session(source, detector_names),
    }


def _combine_config(source: str, detect: dict, ensemble: dict) -> dict:
    """Merge the Detect-step and Ensemble-step pieces into one pipeline config."""
    pipeline_kwargs = {
        "detector_names": detect["detector_names"],
        "detector_params": detect["detector_params"],
        "weighting_scheme": ensemble["weighting_scheme"],
        "ensemble_mode": ensemble["ensemble_mode"],
        "threshold_percentile": ensemble["threshold_percentile"],
        "contamination": ensemble["contamination"],
        "magnitude": ensemble["magnitude"],
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
        ensemble["weighting_scheme"],
        ensemble["threshold_percentile"],
        ensemble["contamination"],
        ensemble["magnitude"],
    )
    return {
        **detect,
        **ensemble,
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
    for house_id in config["houses"]:
        df = load_house_events(source, house_id)
        try:
            result, error = run_house(house_id, df, **config["pipeline_kwargs"])
        except Exception as e:  # noqa: BLE001 - fragile deps (e.g. tick) may raise
            result, error = None, str(e)
        if error:
            st.warning(f"[{house_id}] {error}")
        else:
            new_results[house_id] = result

    st.session_state[f"sig_{source}"] = config["signature"]
    st.session_state[f"results_{source}"] = new_results
    return new_results


# ============================================================================
# RESULT TABS
# ============================================================================
def _render_house_results(source: str, house_view: str, r: Any) -> None:
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
    customdata = list(zip(*[details[c].round(3) for c in score_cols]))

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
    if threshold is not None:
        fig_timeline.add_hline(
            y=threshold,
            line={"color": ANOMALY, "width": 1.5, "dash": "dash"},
            annotation_text=f"threshold (p{getattr(r.ensemble, 'ensemble_threshold_percentile', '?')})",
            annotation_position="top right",
        )
    apply_layout(
        fig_timeline, "Timeline: ensemble scores (red = above threshold)", height=460
    )
    display_chart(fig_timeline, key=f"timeline_{source}_{house_view}")
    st.caption("Red = above the threshold percentile · each point = one day.")

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

    csv = details.to_csv(index=False)
    st.download_button(
        "Download results (CSV)",
        data=csv,
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
    _render_house_results(source, house_view, results[house_view])

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


CLOUD_GRID = 60


def _score_mesh(
    detector: Any, pca: Any, x_range: tuple[float, float], y_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a fitted detector over a mesh of the PCA plane.

    Mesh points are generated in 2-D PCA space, mapped back to the 9-D feature
    space with ``pca.inverse_transform`` and scored — so the background gradient
    is the detector's *actual* score field, exactly like the 2D Playground.
    """
    xs = np.linspace(x_range[0], x_range[1], CLOUD_GRID)
    ys = np.linspace(y_range[0], y_range[1], CLOUD_GRID)
    xx, yy = np.meshgrid(xs, ys)
    mesh_2d = np.column_stack([xx.ravel(), yy.ravel()])
    mesh_full = pca.inverse_transform(mesh_2d)
    _, grid_scores = detector.predict(mesh_full)
    return xx, yy, grid_scores.reshape(xx.shape)


def _build_detector_cloud(
    name: str,
    X_2d: np.ndarray,
    pca: Any,
    scores: np.ndarray,
    detector: Any,
) -> go.Figure:
    """2-D score cloud over the PCA plane, like the 2D Playground.

    A mesh gradient (the detector's own score field over the grid, drawn with
    the same colorscale as the playground) sits under the daily points, which
    are colored by their individual score.
    """
    x_margin = 0.1 * (X_2d[:, 0].max() - X_2d[:, 0].min() + 1e-6)
    y_margin = 0.1 * (X_2d[:, 1].max() - X_2d[:, 1].min() + 1e-6)
    x_range = (X_2d[:, 0].min() - x_margin, X_2d[:, 0].max() + x_margin)
    y_range = (X_2d[:, 1].min() - y_margin, X_2d[:, 1].max() + y_margin)

    _, _, zz = _score_mesh(detector, pca, x_range, y_range)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_range[0], x_range[1], CLOUD_GRID),
            y=np.linspace(y_range[0], y_range[1], CLOUD_GRID),
            z=zz,
            colorscale=[[0, PRIMARY], [1, ANOMALY]],
            showscale=False,
            opacity=0.55,
            contours={"showlines": False},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker={
                "size": 7,
                "color": scores,
                "colorscale": [[0, PRIMARY], [1, ANOMALY]],
                "cmin": 0,
                "cmax": 1,
                "opacity": 0.9,
                "line": {"width": 0.5, "color": "rgba(255,255,255,0.35)"},
                "colorbar": {
                    "title": {"text": "anomaly score", "font": {"size": 11}},
                    "thickness": 12,
                    "len": 0.7,
                },
            },
            hovertemplate="score: %{marker.color:.3f}<extra></extra>",
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

    - ``"detect"`` → one tuneable, clickable card per detector, grouped by family.
    - ``"ensemble"`` → how the selected detectors combine into one anomaly verdict.
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

    config = _combine_config(source, detect, ensemble)
    results = _run_pipeline(source, config)

    _render_anomalies_tab(results, source)
