"""Reusable Streamlit UI building blocks shared by all views."""

import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from references import KIND_LABELS, resources_for
from streamlit_config import CatParam
from theme import (
    AUROC_BAD,
    AUROC_GOOD,
    AUROC_MID,
    MUTED,
    PRIMARY,
    TEXT,
    badge,
    display_chart,
    family_color,
)

logger = logging.getLogger(__name__)


def section_title(text: str, level: str = "###") -> None:
    st.markdown(f"{level} {text}")


def metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of key/value metrics in balanced columns."""
    cols = st.columns(max(1, len(metrics)))
    for col, (label, value) in zip(cols, metrics, strict=True):
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
# Guided workflow stepper - Native Streamlit implementation
# ============================================================================
def guided_stepper(steps: list[str], key: str) -> int:
    """
    Modern guided workflow stepper using native Streamlit components.

    - Tracks max completed step (prevents skipping forward)
    - Only allows: completed steps, or the immediate next step
    - Native buttons with disabled state for locked steps
    - Progress bar at top
    - Returns current step index
    """
    # Initialize state
    st.session_state.setdefault(f"{key}_current", 0)
    st.session_state.setdefault(f"{key}_completed", 0)

    current = st.session_state[f"{key}_current"]
    completed = st.session_state[f"{key}_completed"]
    n = len(steps)

    # Progress bar (replaces complex CSS connector)
    progress_val = int((completed / (n - 1)) * 100) if n > 1 else 100
    st.progress(progress_val)

    # Native buttons in columns
    cols = st.columns(n, gap="small")
    for i, (col, step_name) in enumerate(zip(cols, steps, strict=True)):
        is_active = (i == current)
        is_locked = (i > completed + 1)

        # Determine label with status indicator
        if i < completed or (i == completed and not is_active):
            label = f":material/check: {step_name}"
        elif is_locked:
            label = f":material/lock: {step_name}"
        else:
            label = f"{i + 1}. {step_name}"

        with col:
            if st.button(
                label=label,
                key=f"{key}_step_{i}",
                disabled=is_locked,
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                st.session_state[f"{key}_current"] = i
                st.rerun()

    return st.session_state[f"{key}_current"]


def advance_step(key: str, n_steps: int = 4) -> None:
    """Advance to next step and mark current as completed."""
    cur = st.session_state.get(f"{key}_current", 0)
    if cur < n_steps - 1:
        st.session_state[f"{key}_current"] = cur + 1
        max_completed = max(st.session_state.get(f"{key}_completed", 0), cur + 1)
        st.session_state[f"{key}_completed"] = max_completed


def back_step(key: str) -> None:
    """Go back one step."""
    if st.session_state.get(f"{key}_current", 0) > 0:
        st.session_state[f"{key}_current"] -= 1


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


def info_box(icon: str, title: str, body: str, color: str = TEXT) -> None:
    """Colored in-flow info box (guided explanation, tips, contextual help)."""
    st.markdown(
        f"<div class='info-card' style='color:{color};border-color:{color}55;"
        f"background:{color}11;'>"
        f"<span class='ic-icon'>{icon}</span>"
        f"<div><b>{title}</b><br>{body}</div></div>",
        unsafe_allow_html=True,
    )


def _click_key(key: str) -> str:
    """Streamlit turns a widget key into a sanitized ``st-key-<key>`` class."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "-", key)


def clickable_cards(
    specs: list[dict], key: str, gap: Literal["small", "medium", "large"] = "medium"
) -> str:
    """Card grid whose cards ARE the control - click anywhere on a card to select it.

    Each spec: ``id``, ``icon``, ``title``, ``description``, ``color``, optional
    ``badge`` and optional ``figure`` (preview rendered under the card). The state
    lives in ``session_state[key]``, which is also the returned value.
    """
    current = st.session_state.get(key, specs[0]["id"] if specs else "")

    # Style the card buttons: full width, left aligned, accent top border.
    # Base styles + selected state (kind="primary") styles.
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
            f"button.st-key-{slug}[kind=\"primary\"], "
            f".st-key-{slug} button[kind=\"primary\"] {{"
            f"background:linear-gradient(180deg, {accent}33, {accent}22);"
            f"border-color:{accent}aa; border-top:4px solid {accent};"
            f"box-shadow:0 0 0 2px {accent}44;}}"
            f"button.st-key-{slug} p {{ margin:0; padding:0; }}"
        )
    st.markdown(f"<style>{''.join(card_css)}</style>", unsafe_allow_html=True)

    cols = st.columns(len(specs), gap=gap)
    for col, spec in zip(cols, specs, strict=True):
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
                width="stretch",
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
        key = f"{prefix}_{param.arg_name}"
        values[param.arg_name] = st.session_state.get(key, param.default)
    return values


def render_param_widgets(params, prefix: str, values: dict, help: bool = False) -> None:
    """Render one slider (or selectbox for enum params) per Param.

    ``help=True`` appends the default value as widget help text (used where
    sliders live under a toggle card instead of a full detector card).
    """
    for param in params:
        key = f"{prefix}_{param.arg_name}"
        if isinstance(param, CatParam):
            options = list(param.options)
            st.selectbox(
                param.label,
                options,
                index=options.index(values[param.arg_name]),
                key=key,
                help=f"Default: {param.default}" if help else None,
            )
        else:
            value_type = type(param.default)
            st.slider(
                param.label,
                value_type(param.min_val),
                value_type(param.max_val),
                value_type(values[param.arg_name]),
                step=value_type(param.step),
                key=key,
            )


def render_resources(method: str) -> None:
    """Collapsible 'Learn more' list of curated references for a detector/concept."""
    resources = resources_for(method)
    if not resources:
        return
    with st.expander(":material/menu_book: Learn more", expanded=False):
        for res in resources:
            st.markdown(
                f"**{KIND_LABELS[res.kind]} · [{res.title}]({res.url})**  \n"
                f"<span style='color:{MUTED};font-size:0.85em;'>{res.note}</span>",
                unsafe_allow_html=True,
            )


def detector_card(
    name: str,
    category: str,
    description: str,
    fig: go.Figure,
    auroc: float | None,
    params=(),
    prefix: str = "",
    values: dict | None = None,
    chart_key: str | None = None,
) -> None:
    """Detector card: header with family badge, chart, AUROC pill and param widgets.

    Shared by the 2D Playground grid and the CASAS detector cards so both tracks
    look and behave identically.
    """
    with st.container(border=True):
        head, badge_col = st.columns([3.2, 1])
        with head:
            st.markdown(f"**{name}**")
            st.caption(description)
        with badge_col:
            badge(category, family_color(category))
        display_chart(fig, key=chart_key)
        auroc_pill(auroc)
        render_resources(name)
        if params:
            st.markdown("---")
            render_param_widgets(params, prefix, values or read_param_values(params, prefix))


# ============================================================================
# Decision boundary rendering (shared by 2D Playground and CASAS score map)
# ============================================================================
def render_decision_boundary(
    detector: Any,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    boundary_color: str,
    transform: Callable | None = None,
    grid_resolution: int = 60,
) -> list[go.Scatter]:
    """Extract detector's real decision boundary via marching squares.

    Returns three-layer seam traces: dark edge → family color → white core.
    """
    from mesh import contour_polylines, score_mesh

    _, _, zz, threshold = score_mesh(
        detector, x_range, y_range, grid=grid_resolution, transform=transform
    )
    polylines = contour_polylines(zz, threshold, x_range, y_range)
    traces = []
    if polylines:
        for line_color, width in [
            ("rgba(8, 12, 20, 0.85)", 5.0),
            (boundary_color, 3.0),
            ("rgba(255, 255, 255, 0.95)", 1.2),
        ]:
            bx, by = [], []
            for pl in polylines:
                bx.extend(pl[:, 0].tolist())
                by.extend(pl[:, 1].tolist())
                bx.append(None)
                by.append(None)
            traces.append(go.Scatter(
                x=bx, y=by, mode="lines",
                line={"color": line_color, "width": width},
                hoverinfo="skip", showlegend=False
            ))
    return traces


# ============================================================================
# Safe detector fit/predict with fallback
# ============================================================================
def safe_fit_predict(
    detector: Any,
    X_train: np.ndarray,
    X_score: np.ndarray,
    fallback_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit detector on X_train, predict on X_score. Returns zeros on failure."""
    try:
        detector.fit(X_train)
        return detector.predict(X_score)
    except Exception as e:
        logger.warning("Detector %s failed: %s", type(detector).__name__, e)
        return (
            np.zeros(len(X_score), dtype=int),
            np.full(len(X_score), fallback_score)
        )
