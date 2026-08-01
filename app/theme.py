"""Shared Plotly theme for a consistent, professional look across all views."""

import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------------
PRIMARY = "#2563eb"
PRIMARY_SOFT = "rgba(37, 99, 235, 0.6)"
ANOMALY = "#dc2626"
ANOMALY_SOFT = "rgba(220, 38, 38, 0.75)"
NEUTRAL = "#64748b"
TEXT = "#1f2937"
MUTED = "#6b7280"
BACKGROUND = "rgba(248, 249, 250, 1)"

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Standard chart heights
HEIGHT_SMALL = 300
HEIGHT_MEDIUM = 380
HEIGHT_LARGE = 460


def apply_layout(
    fig: go.Figure,
    title: str | None = None,
    height: int = HEIGHT_MEDIUM,
    margins: dict | None = None,
) -> go.Figure:
    """Apply the shared base layout (template, fonts, colors, centered title)."""
    layout = {
        "template": PLOTLY_TEMPLATE,
        "font": {"color": TEXT},
        "paper_bgcolor": BACKGROUND,
        "plot_bgcolor": BACKGROUND,
        "margin": {"l": 40, "r": 20, "t": 45, "b": 40, **(margins or {})},
        "height": height,
    }
    if title:
        layout["title"] = {
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": TEXT},
            "x": 0.5,
            "xanchor": "center",
        }
    fig.update_layout(**layout)
    return fig


def display_chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the app-wide theme and toolbar settings."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )
