"""Reusable Streamlit UI building blocks shared by all views."""

import plotly.graph_objects as go
import streamlit as st
from theme import (
    ANOMALY,
    ANOMALY_SOFT,
    AUROC_BAD,
    AUROC_GOOD,
    AUROC_MID,
    ERROR,
    MUTED,
    PRIMARY,
    SUCCESS,
    TEXT,
    WARNING,
    apply_layout,
    badge,
    display_chart,
    family_color,
)


def section_title(text: str, level: str = "###") -> None:
    st.markdown(f"{level} {text}")


def metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of key/value metrics in balanced columns."""
    cols = st.columns(max(1, len(metrics)))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def chart_pair(
    fig_a: go.Figure,
    fig_b: go.Figure,
    key_a: str | None = None,
    key_b: str | None = None,
) -> None:
    """Render two themed charts side by side."""
    col1, col2 = st.columns(2)
    with col1:
        display_chart(fig_a, key=key_a)
    with col2:
        display_chart(fig_b, key=key_b)


# ============================================================================
# Guided workflow
# ============================================================================
def page_header(
    title: str, subtitle: str | None = None, breadcrumb: str | None = None
) -> None:
    """Consistent page header with an optional workflow breadcrumb."""
    if breadcrumb:
        st.caption(breadcrumb)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def guided_stepper(steps: list[str], current: int | str, key: str) -> int:
    """Guided workflow: clickable step tabs to move between the 3 stages.

    Each step renders as a tab button and stores its index in ``session_state[key]``.
    ``current`` may be an index (first run) or a stored label.
    """
    current_index = current if isinstance(current, int) else current

    cols = st.columns(len(steps), gap="small")
    for i, col in enumerate(cols):
        active = i == current_index
        label = f"{i + 1} · {steps[i]}"
        with col:
            if st.button(
                label,
                key=f"{key}_tab_{i}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state[key] = i
                st.rerun()
    return _read_stepper_selection(steps, key)


def _read_stepper_selection(steps: list[str], key: str) -> int:
    """Resolve which step is active after the tabs (or a Next/Back jump)."""
    return st.session_state.get(key, 0)


def breadcrumb(parts: list[tuple[str, bool]], context: str | None = None) -> None:
    """Contextual breadcrumb: ``parts`` = [(label, is_current), ...]."""
    html = ['<div class="breadcrumb">']
    for i, (label, is_current) in enumerate(parts):
        cls = "crumb-current" if is_current else ""
        html.append(f'<span class="crumb {cls}">{label}</span>')
        if i < len(parts) - 1:
            html.append('<span class="crumb-sep">›</span>')
    if context:
        html.append(f'<span class="crumb-context">{context}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def family_header(name: str, category: str, description: str = "") -> None:
    """Colored family block header: left rail, family name, description."""
    color = family_color(category)
    st.markdown(
        f'<div class="family-header" style="color:{color};">'
        f'<span class="family-name">{name}</span>'
        f'<span class="family-desc">{description}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Headers, section helpers and info/stat cards
# ============================================================================
def colored_section_header(number: str, title: str, color: str, hint: str = "") -> None:
    """Main-area colored stage header: numbered badge + title + optional hint."""
    st.markdown(
        f"<div class='sb-title' style='margin-top:18px;'><span class='sb-num' "
        f"style='color:{color};background:{color}22;border:1px solid {color}55;'>"
        f"{number}</span><span>{title}</span></div>",
        unsafe_allow_html=True,
    )
    if hint:
        st.caption(hint)


def sb_section_header(number: str, title: str, color: str, expanded: bool = True):
    """Sidebar numbered section header + expander body (returns the expander)."""
    st.sidebar.markdown(
        f"<div class='sb-title' style='margin-top:10px;'><span class='sb-num' "
        f"style='color:{color};background:{color}22;border:1px solid {color}55;'>"
        f"{number}</span><span>{title}</span></div>",
        unsafe_allow_html=True,
    )
    return st.sidebar.expander("", expanded=expanded)


def info_box(icon: str, title: str, body: str, color: str = TEXT) -> None:
    """Colored in-flow info box (guided explanation, tips, contextual help)."""
    st.markdown(
        f"<div class='info-card' style='color:{color};border-color:{color}55;"
        f"background:{color}11;'>"
        f"<span class='ic-icon'>{icon}</span>"
        f"<div><b>{title}</b><br>{body}</div></div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, color: str = TEXT, sub: str = "") -> None:
    """Single large stat card (KPI / house AUROC); accent color on the top rail."""
    sub_html = f"<div class='stat-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='stat-card' style='--card-accent:{color};'>"
        f"<div class='stat-label'>{label}</div>"
        f"<div class='stat-value' style='color:{color};'>{value}</div>"
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def interpret_auroc(auroc: float | None) -> tuple[str, str]:
    """Semantic verdict for an AUROC value: (color, human label)."""
    if auroc is None:
        return MUTED, "not measured"
    if auroc > 0.75:
        return SUCCESS, "✓ Good (>0.75)"
    if auroc > 0.7:
        return WARNING, "~ Fair (0.70–0.75)"
    return ERROR, "✗ Poor (<0.70) — try tuning"


def param_chips(params, values: dict) -> None:
    """Read-only parameter chips: ``label: value`` (advanced cards summary)."""
    chips = "".join(
        f"<span class='param-chip'>{p.label}: <b>{values.get(p.kwarg, p.default)}</b></span>"
        for p in params
    )
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


def _click_key(key: str) -> str:
    """Streamlit turns a widget key into a sanitized ``st-key-<key>`` class."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "-", key)


def clickable_cards(specs: list[dict], key: str, gap: str = "medium") -> str:
    """Card grid whose cards ARE the control — click anywhere on a card to select it.

    Each spec: ``id``, ``icon``, ``title``, ``description``, ``color``, optional
    ``badge`` and optional ``figure`` (preview rendered under the card). The state
    lives in ``session_state[key]``, which is also the returned value.
    """
    current = st.session_state.get(key, specs[0]["id"] if specs else "")

    # Style the card buttons: full width, left aligned, accent top border.
    card_css = []
    for spec in specs:
        slug = _click_key(f"{key}_card_{spec['id']}")
        accent = spec.get("color", PRIMARY)
        card_css.append(
            f"button.st-key-{slug}, .st-key-{slug} button {{"
            f"text-align:left; justify-content:flex-start; height:auto;"
            f"padding:14px 16px; border-radius:10px;"
            f"border:1px solid {accent}55; border-top:4px solid {accent};"
            f"background:linear-gradient(180deg, {accent}1c, var(--bg-surface));"
            f"white-space:normal; font-weight:400; line-height:1.5; }}"
            f"button.st-key-{slug} p {{ margin:0; padding:0; }}"
        )
    st.markdown(f"<style>{''.join(card_css)}</style>", unsafe_allow_html=True)

    cols = st.columns(len(specs), gap=gap)
    for col, spec in zip(cols, specs):
        with col:
            selected = spec["id"] == current
            lines = [f"{spec.get('icon', '')} **{spec['title']}**"]
            if spec.get("description"):
                lines.append(spec["description"])
            if spec.get("badge"):
                lines.append(f"_{spec['badge']}_")
            st.button(
                "\n\n".join(lines),
                key=f"{key}_card_{spec['id']}",
                use_container_width=True,
                type="primary" if selected else "secondary",
                on_click=_click_set,
                args=(key, spec["id"]),
            )
            if spec.get("figure") is not None:
                display_chart(spec["figure"], key=f"{key}_{spec['id']}_preview")
    return current


def _click_set(key: str, value: str) -> None:
    """Button callback: store the clicked card id into session state."""
    st.session_state[key] = value


# ============================================================================
# Synthetic temporal patterns (chosen in the Data step, used by the Features)
# ============================================================================
PATTERN_EXPLANATIONS: dict[str, str] = {
    "regular": (
        "Steady rhythm: events arrive at a nearly constant pace, gaps are all similar "
        "(**low CV / Fano factor**)."
    ),
    "bursty": (
        "Activity clusters into short bursts with long silences between them "
        "(**high CV / Fano factor**)."
    ),
    "day_night": (
        "Active by day (08–22h), almost no nocturnal events — strong day/night "
        "contrast."
    ),
}


def pattern_preview(pattern: str) -> go.Figure:
    """One-day activity profile of a synthetic temporal pattern.

    Hourly event count computed from the generator, night hours shaded — a
    clear one-glance read of what each pattern (regular/bursty/day_night) looks
    like. Used as the preview inside the picker cards (Data step).
    """
    import numpy as np
    import pandas as pd

    from features.event_driven_extractors import generate_synthetic_events

    df = generate_synthetic_events(
        n_days=1, pattern=pattern, n_sensors=3, events_per_day=80, seed=42
    )
    ts = pd.to_datetime(df["timestamp"])
    hours = ts.dt.hour + ts.dt.minute / 60.0
    counts, _ = np.histogram(hours, bins=24, range=(0, 24))

    fig = go.Figure(
        go.Bar(
            x=np.arange(24) + 0.5,
            y=counts,
            marker_color=PRIMARY,
            marker_line_width=0,
            hoverinfo="skip",
        )
    )
    fig.add_vrect(x0=0, x1=8, fillcolor=ANOMALY_SOFT, line_width=0)
    fig.add_vrect(x0=22, x1=24, fillcolor=ANOMALY_SOFT, line_width=0)
    apply_layout(fig, None, height=135)
    fig.update_layout(
        margin={"l": 6, "r": 6, "t": 8, "b": 28},
        bargap=0.15,
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="Hour of day",
        showgrid=False,
        zeroline=False,
        tickvals=[0, 6, 12, 18, 23],
    )
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


# ============================================================================
# Detector strengths / weaknesses (shared by cards + Info tab)
# ============================================================================
DETECTOR_STRENGTHS: dict[str, tuple[str, str]] = {
    "Isolation Forest": (
        "Scalable, no distribution assumption, robust in higher dimensions",
        "Struggles with strongly correlated features",
    ),
    "Extended IForest": (
        "Oblique splits handle rotated / high-dimensional data",
        "Less battle-tested than the classic isolation forest",
    ),
    "Mahalanobis": (
        "Covariance-aware distance, single scalar threshold",
        "Assumes Gaussian data; sensitive to outliers in the training set",
    ),
    "Elliptic Envelope": (
        "Robust MCD fit resists outliers during training",
        "Gaussian-ellipsoid assumption; heavy on large datasets",
    ),
    "Robust Covariance": (
        "Very robust covariance estimate (MCD)",
        "Elliptical-shape assumption; can be slow",
    ),
    "KNN": (
        "Distribution-agnostic, no density or shape assumption",
        "Slow on large data; K is a sensitive hyperparameter",
    ),
    "OC-SVM": (
        "Flexible non-convex boundary with kernels",
        "Expensive training; nu is delicate to tune",
    ),
    "LOF": (
        "Catches local deviations even inside dense regions",
        "O(n²) behaviour; neighbourhood size affects results strongly",
    ),
    "Z-Score": (
        "Dead simple, interpretable per-feature distance",
        "Assumes roughly Gaussian features; ignores correlations",
    ),
    "PCA Reconstruction": (
        "Linear subspace structure; cheap prediction once fitted",
        "Misses non-linear structure; component count to tune",
    ),
    "HMM": (
        "Models temporal regimes and regime transitions",
        "Gaussian-per-state assumption; slower training",
    ),
    "Hawkes": (
        "Self-exciting point-process view of the event stream",
        "Heavy native dependency; score is illustrative rather than exact",
    ),
}


def strength_box(name: str) -> None:
    """Small strengths / weaknesses box per detector card."""
    text = DETECTOR_STRENGTHS.get(name)
    if not text:
        return
    strengths, weaknesses = text
    st.markdown(
        f"<div class='info-card' style='color:{MUTED};border-color:{MUTED}44;"
        f"background:rgba(255,255,255,0.02);'>"
        f"<span class='ic-icon'>📌</span><div>"
        f"<b>Strengths:</b> {strengths}<br>"
        f"<b style='color:{ANOMALY};'>Weaknesses:</b> {weaknesses}</div></div>",
        unsafe_allow_html=True,
    )


def auroc_pill(auroc: float | None, label: str = "AUROC") -> None:
    """Colored pill for an AUROC value with an interpretive verdict."""
    if auroc is None:
        st.markdown(
            f"<span class='auroc-pill' style='background:{MUTED}22;color:{MUTED};"
            f"border:1px solid {MUTED}66;'>{label}: not measured</span>",
            unsafe_allow_html=True,
        )
        return
    if auroc >= 0.8:
        color, verdict = AUROC_GOOD, "Excellent"
    elif auroc >= 0.7:
        color, verdict = AUROC_MID, "Marginal"
    else:
        color, verdict = AUROC_BAD, "Poor"
    st.markdown(
        f"<span class='auroc-pill' style='background:{color}1e;color:{color};"
        f"border:1px solid {color}88;'>{label}: {auroc:.3f} · {verdict}</span>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Detector parameter widgets
# ============================================================================
def read_param_values(params, prefix: str) -> dict:
    """Read the current widget values for a detector's params from session state."""
    values = {}
    for param in params:
        key = f"{prefix}_{param.kwarg}"
        values[param.kwarg] = st.session_state.get(key, param.default)
    return values


def render_param_widgets(params, prefix: str, values: dict) -> None:
    """Render one slider (or selectbox for enum params) per ParamSpec."""
    for param in params:
        key = f"{prefix}_{param.kwarg}"
        if param.options:
            options = list(param.options)
            st.selectbox(
                param.label,
                options,
                index=options.index(values[param.kwarg]),
                key=key,
            )
        else:
            value_type = type(param.default)
            st.slider(
                param.label,
                value_type(param.min),
                value_type(param.max),
                value_type(values[param.kwarg]),
                step=value_type(param.step),
                key=key,
            )


def detector_card(
    name: str,
    category: str,
    description: str,
    fig: go.Figure,
    auroc: float | None,
    params=(),
    prefix: str = "",
    show_params: bool = True,
    values: dict | None = None,
    chart_key: str | None = None,
    params_readonly: bool = False,
    show_strengths: bool = False,
) -> None:
    """Detector card: header with family badge, chart, AUROC pill and params.

    Shared by the 2D Playground grid and the CASAS detector cards so both tracks
    look and behave identically. ``params_readonly`` renders the params as chips
    (values driven by the sidebar); otherwise the widgets themselves are drawn.
    """
    with st.container(border=True):
        head, badge_col = st.columns([3.2, 1])
        with head:
            st.markdown(f"**{name}**")
            if show_strengths:
                st.caption(description, unsafe_allow_html=False)
            else:
                st.caption(description)
        with badge_col:
            badge(category, family_color(category))
        display_chart(fig, key=chart_key)
        auroc_pill(auroc)
        if show_strengths:
            strength_box(name)
        if show_params and params:
            if params_readonly:
                with st.expander("Parameters"):
                    values = values or read_param_values(params, prefix)
                    param_chips(params, values)
            else:
                st.markdown("---")
                render_param_widgets(
                    params, prefix, values or read_param_values(params, prefix)
                )
