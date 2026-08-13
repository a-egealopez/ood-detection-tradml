"""Shared Plotly theme: design tokens, family colors and card/badge helpers.

Grounds the look in the subject: smart-home sensors and time-series map to warm
indigo ("circuits") plus digital green ("readouts"). Everything graphic the UI
draws reads from these tokens so a style change is a single-source edit.
"""

import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Core palette (dark canvas + warm indigo accent / digital green success)
# ----------------------------------------------------------------------------
BG_CANVAS = "#0f1419"  # very dark, near-black app background
BG_SURFACE = "#1a1f2e"  # elevated surfaces (cards, sidebar, panels)
TEXT = "#f0f4f9"  # warm white primary text
MUTED = "#9ca3af"  # secondary / captions
BORDER = "#374151"  # subtle borders and gridlines

PRIMARY = "#3b82f6"  # warm indigo (action / focus)
PRIMARY_SOFT = "rgba(59, 130, 246, 0.45)"  # translucent indigo for markers/areas
SUCCESS = "#10b981"  # digital green (AUROC >= 0.8)
WARNING = "#f59e0b"  # sensor orange (0.7-0.8)
ERROR = "#ef4444"  # light red (< 0.7)
ANOMALY = "#ec4899"  # magnetic pink (anomalous points / alerts)
ANOMALY_SOFT = "rgba(236, 72, 153, 0.55)"

# ----------------------------------------------------------------------------
# Family colors: one accent per detector category (badges + section headers)
# ----------------------------------------------------------------------------
FAMILY_DENSITY = "#14b8a6"
FAMILY_GAUSSIAN = "#6366f1"
FAMILY_DISTANCE = "#8b5cf6"
FAMILY_BOUNDARY = "#f59e0b"
FAMILY_UNIVARIATE = "#6b7280"
FAMILY_DIMENSIONALITY = "#06b6d4"
FAMILY_SEQUENTIAL = "#f43f5e"

FAMILY_COLORS = {
    "Density": FAMILY_DENSITY,
    "Gaussian": FAMILY_GAUSSIAN,
    "Distance": FAMILY_DISTANCE,
    "One-Class SVM": FAMILY_BOUNDARY,
    "Univariate": FAMILY_UNIVARIATE,
    "Dimensionality": FAMILY_DIMENSIONALITY,
    "Sequential": FAMILY_SEQUENTIAL,
}

# AUROC quality thresholds
AUROC_GOOD = SUCCESS
AUROC_MID = WARNING
AUROC_BAD = ERROR

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Standard chart heights
HEIGHT_SMALL = 300
HEIGHT_MEDIUM = 380
HEIGHT_LARGE = 460


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def family_color(category: str) -> str:
    """Accent color for a detector category (falls back to PRIMARY)."""
    return FAMILY_COLORS.get(category, PRIMARY)


def auroc_color(auroc: float | None) -> str:
    """Semantic color for an AUROC value: good / acceptable / poor."""
    if auroc is None:
        return MUTED
    if auroc >= 0.80:
        return AUROC_GOOD
    if auroc >= 0.70:
        return AUROC_MID
    return AUROC_BAD


def badge(text: str, color: str, size: str = "0.72em") -> None:
    """Render a small colored pill (used for family badges and AUROC)."""
    import streamlit as st

    st.markdown(
        f"<span style='background:{color}22;color:{color};padding:2px 9px;"
        f"border-radius:999px;font-size:{size};font-weight:600;"
        f"border:1px solid {color}55;white-space:nowrap;'>{text}</span>",
        unsafe_allow_html=True,
    )


def apply_layout(
    fig: go.Figure,
    title: str | None = None,
    height: int = HEIGHT_MEDIUM,
    margins: dict | None = None,
) -> go.Figure:
    """Apply the shared dark layout (background, fonts, grid, centered title)."""
    layout = {
        "font": {"color": TEXT, "size": 12, "family": "Segoe UI, sans-serif"},
        "paper_bgcolor": BG_SURFACE,
        "plot_bgcolor": BG_CANVAS,
        "xaxis": {"gridcolor": BORDER, "linecolor": BORDER},
        "yaxis": {"gridcolor": BORDER, "linecolor": BORDER},
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
    import streamlit as st

    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )


# ----------------------------------------------------------------------------
# Global CSS (injected once from streamlit_app) - signature elements
# ----------------------------------------------------------------------------
BASE_CSS = """
<style>
/* layout surfaces */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] { background-color: var(--bg-canvas); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1d2433 0%, var(--bg-surface) 100%);
  border-right: 1px solid var(--border);
}
h1, h2, h3, h4 { color: var(--text-primary); letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"] { color: var(--text-muted); }

/* Stage cards (feature-extraction picker): colored left rail + tinted body */
.stage-card {
  display: flex; flex-direction: column; gap: 8px;
  padding: 14px 16px; border-radius: 8px;
  border: 1px solid var(--border); border-left: 4px solid currentColor;
  background: linear-gradient(90deg, rgba(59,130,246,0.06), transparent);
}
.stage-card .stage-head { display: flex; align-items: center; gap: 10px; }
.stage-card .stage-title { font-size: 15px; font-weight: 700; }
.stage-card .stage-desc { font-size: 13px; color: var(--text-muted); line-height: 1.45; }
.stage-badge {
  display: inline-flex; align-items: center; gap: 6px; align-self: flex-start;
  padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
  border: 1px solid; letter-spacing: 0.03em;
}
.stage-badge .glyph { font-size: 12px; }

/* Family section header: colored left rail + family color text */
.family-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; margin: 8px 0 16px 0;
  border-left: 4px solid currentColor;
  background: linear-gradient(90deg, rgba(59,130,246,0.08), transparent);
  border-radius: 6px;
}
.family-header .family-name {
  font-size: 18px; font-weight: 600; letter-spacing: 0.02em;
}
.family-header .family-desc { font-size: 13px; color: var(--text-muted); }

/* AUROC pill (animated on appear) */
.auroc-pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 12px; border-radius: 999px; margin-top: 10px;
  font-size: 12px; font-weight: 700; letter-spacing: 0.04em;
  animation: fade-pop 0.4s ease;
}
@keyframes fade-pop {
  from { opacity: 0; transform: translateY(3px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Sidebar: sections as numbered cards */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { margin-top: 0; }
.sb-title {
  display: flex; align-items: center; gap: 10px;
  margin: 6px 0 2px 0; font-size: 15px; font-weight: 700;
  color: var(--text-primary); letter-spacing: 0.02em;
}
.sb-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 8px; flex: 0 0 auto;
  font-size: 14px; font-weight: 800;
}
.sb-expander { margin-top: 2px; }
.sb-db {
  display: inline-block; padding: 2px 8px; margin-top: 2px;
  border-radius: 6px; font-size: 11px; color: var(--text-muted);
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  font-family: 'SF Mono', 'Menlo', monospace;
}
.sb-note { font-size: 12px; color: var(--text-muted); }

/* Status bar with pulsing dot */
.sb-status {
  display: flex; align-items: center; gap: 8px; margin-top: 14px;
  padding: 9px 12px; border-radius: 8px; font-size: 13px;
  border: 1px solid var(--border);
}
.sb-status .dot {
  width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto;
  background: currentColor;
}
.sb-status.computing .dot { animation: pulse 1.1s ease-in-out infinite; }
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.35; transform: scale(0.8); }
}

/* Breadcrumb chain */
.breadcrumb {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  font-size: 13px; margin-bottom: 16px; color: var(--text-muted);
}
.breadcrumb .crumb { display: inline-flex; align-items: center; gap: 8px; }
.breadcrumb .crumb-sep { color: var(--border); }
.breadcrumb .crumb-current { font-weight: 700; color: var(--action-primary); }
.breadcrumb .crumb-context {
  display: inline-flex; align-items: center; gap: 10px; width: 100%;
  margin-top: 4px; font-size: 12px; color: var(--text-muted);
  flex-wrap: wrap;
}

/* Selectable card grid (data source, extraction methods) */
.select-card {
  display: flex; flex-direction: column; gap: 8px; min-height: 100%;
  padding: 14px 16px; border-radius: 10px;
  border: 1px solid var(--border); border-top: 4px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent);
  transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}
.select-card .select-head { display: flex; align-items: center; gap: 10px; }
.select-card .select-icon { font-size: 22px; line-height: 1; }
.select-card .select-title { font-size: 15px; font-weight: 700; letter-spacing: 0.01em; }
.select-card .select-desc { font-size: 13px; color: var(--text-muted); line-height: 1.45; }
.select-card .select-meta { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 2px; }
.select-card.selected {
  border-color: var(--card-accent, var(--action-primary));
  border-top-color: var(--card-accent, var(--action-primary));
  background: linear-gradient(180deg, var(--card-accent, transparent) 0%, rgba(255,255,255,0.02) 100%);
  box-shadow: 0 0 0 1px var(--card-accent, var(--action-primary)) inset,
              0 4px 18px rgba(59, 130, 246, 0.16);
}

/* Colored info box (guided mode, tips, warnings in-line) */
.info-card {
  display: flex; gap: 10px; align-items: flex-start;
  padding: 10px 14px; border-radius: 8px; margin: 8px 0;
  border: 1px solid; font-size: 13px; line-height: 1.5;
}
.info-card .ic-icon { font-size: 16px; line-height: 1.3; flex: 0 0 auto; }

/* Stat metric card (KPIs, house AUROC cards) */
.stat-card {
  padding: 14px 16px; border-radius: 10px; text-align: center;
  border: 1px solid var(--border); background: var(--bg-surface);
  border-top: 3px solid var(--card-accent, var(--action-primary));
}
.stat-card .stat-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-muted); margin-bottom: 4px;
}
.stat-card .stat-value {
  font-size: 26px; font-weight: 800; line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.stat-card .stat-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

/* Read-only parameter chips (advanced cards) */
.param-chip {
  display: inline-flex; align-items: baseline; gap: 6px;
  padding: 2px 9px; margin: 2px 4px 2px 0; border-radius: 6px;
  font-size: 11px; font-family: 'SF Mono', 'Menlo', monospace;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
}
.param-chip b { color: var(--text-primary); }

/* Small responsive tweaks for narrow widths */
@media (max-width: 768px) {
  .family-header { flex-direction: column; align-items: flex-start; gap: 4px; }
  .sb-db { font-size: 10px; }
  [data-testid="stSlider"] [role="slider"] { min-height: 34px; min-width: 34px; }
}
</style>
"""


def _root_vars() -> str:
    """Expose the Python design tokens as CSS custom properties.

    ``BASE_CSS`` references ``var(--bg-canvas)``, ``var(--action-primary)``, etc.;
    Streamlit only defines its own ``--primary-color`` family, so these must be
    declared here — keeping ``theme.py`` the single source for both Python and CSS.
    """
    return (
        ":root {"
        f"--bg-canvas: {BG_CANVAS};"
        f"--bg-surface: {BG_SURFACE};"
        f"--text-primary: {TEXT};"
        f"--text-muted: {MUTED};"
        f"--border: {BORDER};"
        f"--action-primary: {PRIMARY};"
        "}"
    )


def inject_theme() -> None:
    """Inject the global CSS (call once, right after set_page_config)."""
    import streamlit as st

    st.markdown(
        f"<style>{_root_vars()}\n{BASE_CSS}</style>", unsafe_allow_html=True
    )
