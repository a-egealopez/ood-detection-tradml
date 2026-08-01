"""Reusable Streamlit UI building blocks shared by all views."""

import plotly.graph_objects as go
import streamlit as st

from theme import display_chart


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
