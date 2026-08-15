"""Feature extraction tutorial: from an event stream to a feature vector.

Single-screen flow (mirrors the Detect step). The data origin is decided in the
main Data step (CASAS track) and reused here. The user picks an extraction method;
a didactic schematic plus a written explanation appear on selection, and the
inspector below turns raw events into the numeric vector for that method.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import itertools

from components import (
    breadcrumb,
    chart_pair,
    clickable_cards,
    colored_section_header,
    render_resources,
)
from data_access import load_all_events
from theme import (
    ANOMALY,
    ANOMALY_SOFT,
    FAMILY_BOUNDARY,
    FAMILY_DENSITY,
    FAMILY_DISTANCE,
    PRIMARY_SOFT,
    TEXT,
    apply_layout,
    display_chart,
)

from features.event_driven_extractors import (
    IntervalStatisticsExtractor,
    NextEventTransitionExtractor,
    NGramTransitionExtractor,
    WindowAggregationExtractor,
    generate_synthetic_events,
)

METHOD_DESCRIPTIONS = {
    "Window Aggregation": (
        "Groups the events of each day and summarizes their hourly distribution into "
        "statistics (counts, entropy, night share). The classic approach in activity "
        "recognition on binary sensors."
    ),
    "Inter-Event Interval (IEI)": (
        "Analyzes the time between consecutive events as a point process. The "
        "coefficient of variation (CV) and the Fano factor distinguish regular, "
        "Poisson-like or 'bursty' processes."
    ),
    "N-gram Transition (Markov)": (
        "Converts the sequence of triggered sensors into a first-order Markov chain. "
        "The entropy of the transition matrix measures how predictable the event "
        "sequence is."
    ),
    "Next-Event Prediction (Markov)": (
        "Learns the transition probabilities of the normal sensor sequence, then "
        "scores each real transition by its likelihood. A very unlikely next sensor "
        "flags a single-event anomaly (DeepLog / n-gram baseline)."
    ),
}

METHOD_GLYPHS = {
    "Window Aggregation": "◧",
    "Inter-Event Interval (IEI)": "≋",
    "N-gram Transition (Markov)": "⇄",
    "Next-Event Prediction (Markov)": "➤",
}

METHOD_COLORS = {
    "Window Aggregation": FAMILY_DENSITY,
    "Inter-Event Interval (IEI)": FAMILY_DISTANCE,
    "N-gram Transition (Markov)": FAMILY_BOUNDARY,
    "Next-Event Prediction (Markov)": FAMILY_BOUNDARY,
}


# ============================================================================
# Didactic schematics: what each extractor "sees" on the raw stream
# ============================================================================
def _schematic_window() -> go.Figure:
    """A day as 24 hourly bins collapsing into a summary feature vector."""
    fig = go.Figure()
    hours = np.arange(24)
    counts = np.array(
        [1, 0, 0, 0, 0, 2, 5, 8, 6, 3, 4, 2, 3, 5, 4, 6, 9, 7, 4, 3, 2, 1, 0, 0],
        dtype=float,
    )
    fig.add_vrect(x0=-0.5, x1=8.5, fillcolor=ANOMALY_SOFT, line_width=0)
    fig.add_vrect(x0=21.5, x1=24.5, fillcolor=ANOMALY_SOFT, line_width=0)
    fig.add_trace(
        go.Bar(x=hours, y=counts, marker_color=PRIMARY_SOFT, hoverinfo="skip")
    )
    fig.add_annotation(
        x=8.5,
        y=11,
        text="night",
        showarrow=False,
        font={"color": "rgba(236,72,153,0.9)", "size": 11},
    )
    fig.add_annotation(
        x=16,
        y=1,
        text="24 hourly counts  →  n_events · entropy · night_share",
        showarrow=False,
        font={"color": TEXT, "size": 11},
    )
    apply_layout(fig, None, height=220)
    fig.update_xaxes(showgrid=False, zeroline=False, title_text="Hour of day")
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


def _schematic_iei() -> go.Figure:
    """Consecutive events with their gaps (intervals), then an interval histogram."""
    t = np.array([0, 1.2, 2.1, 4.4, 5.0, 8.3])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=[0] * len(t),
            mode="markers",
            marker={
                "size": 10,
                "color": PRIMARY_SOFT,
                "symbol": "line-ns",
                "line": {"width": 2, "color": PRIMARY_SOFT},
            },
            hoverinfo="skip",
        )
    )
    for t0, t1 in itertools.pairwise(t):
        fig.add_shape(
            type="line",
            x0=t0,
            x1=t1,
            y0=0.25,
            y1=0.25,
            line={"color": ANOMALY, "width": 1.5, "dash": "dot"},
        )
        fig.add_annotation(
            x=(t0 + t1) / 2,
            y=0.5,
            text="Δt",
            showarrow=False,
            font={"color": ANOMALY, "size": 10},
        )
    fig.add_annotation(
        x=4.5,
        y=-0.9,
        text="gaps Δt  →  CV & Fano factor",
        showarrow=False,
        font={"color": TEXT, "size": 11},
    )
    apply_layout(fig, None, height=220)
    fig.update_xaxes(showgrid=False, zeroline=False, title_text="Time")
    fig.update_yaxes(visible=False, range=[-1.2, 1])
    return fig


def _schematic_markov() -> go.Figure:
    """A sensor sequence with transitions, feeding a transition matrix."""
    seq = "A B A C B C A"
    tokens = seq.split()
    n = len(tokens)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[0] * n,
            mode="markers+text",
            marker={"size": 16, "color": PRIMARY_SOFT},
            text=tokens,
            textposition="top center",
            textfont={"size": 11, "color": TEXT},
            hoverinfo="skip",
        )
    )
    for i in range(n - 1):
        fig.add_annotation(
            x=i + 1,
            y=0,
            ax=i,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor=ANOMALY,
        )
    fig.add_annotation(
        x=(n - 1) / 2,
        y=-1.1,
        text="transitions A→B  →  Markov matrix entropy",
        showarrow=False,
        font={"color": TEXT, "size": 11},
    )
    apply_layout(fig, None, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-1.4, 1])
    return fig


def _schematic_next_event() -> go.Figure:
    """A transition with high vs. low likelihood under the learned model."""
    seq = "A B A B"
    tokens = seq.split()
    n = len(tokens)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[0] * n,
            mode="markers+text",
            marker={"size": 16, "color": PRIMARY_SOFT},
            text=tokens,
            textposition="top center",
            textfont={"size": 11, "color": TEXT},
            hoverinfo="skip",
        )
    )
    for i in range(n - 1):
        color = ANOMALY if i == 2 else PRIMARY_SOFT
        fig.add_annotation(
            x=i + 1,
            y=0,
            ax=i,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor=color,
        )
    fig.add_annotation(
        x=1,
        y=-1.1,
        text="A→B likely (P high)",
        showarrow=False,
        font={"color": PRIMARY_SOFT, "size": 10},
    )
    fig.add_annotation(
        x=3,
        y=-1.1,
        text="B→A unlikely (P low)  →  anomaly",
        showarrow=False,
        font={"color": ANOMALY, "size": 10},
    )
    apply_layout(fig, None, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-1.6, 1])
    return fig


def _method_schematic(method: str) -> go.Figure:
    if method == "Window Aggregation":
        return _schematic_window()
    if method == "Inter-Event Interval (IEI)":
        return _schematic_iei()
    if method == "N-gram Transition (Markov)":
        return _schematic_markov()
    return _schematic_next_event()


# Short card copy for the method picker grid.
METHOD_CARDS = {
    "Window Aggregation": {
        "desc": "Daily hour-by-hour summary (counts, entropy)",
        "badge": "Detects: Contextual + Collective",
    },
    "Inter-Event Interval (IEI)": {
        "desc": "Analyze gaps between events (regularity, rhythm)",
        "badge": "Detects: Point + Collective",
    },
    "N-gram Transition (Markov)": {
        "desc": "Sensor sequence patterns (Markov chain)",
        "badge": "Detects: Collective sequence",
    },
    "Next-Event Prediction (Markov)": {
        "desc": "Score each transition by its likelihood",
        "badge": "Detects: Point + Collective sequence",
    },
}


# ============================================================================
# Inspector helpers (synthetic pattern picker lives with the Data step)
# ============================================================================
def _plot_window_aggregation(
    group: pd.DataFrame, diag: dict
) -> tuple[go.Figure, go.Figure]:
    group = group.copy()
    group["timestamp"] = pd.to_datetime(group["timestamp"])
    sensors = sorted(group["sensor_id"].unique())

    timeline = go.Figure()
    for sensor in sensors:
        sub = group[group["sensor_id"] == sensor]
        timeline.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=[sensor] * len(sub),
                mode="markers",
                marker={"size": 8, "color": PRIMARY_SOFT},
                hoverinfo="skip",
            )
        )
    day_start = group["timestamp"].dt.normalize().iloc[0]
    timeline.add_vrect(
        x0=day_start,
        x1=day_start + pd.Timedelta(hours=8),
        fillcolor=ANOMALY_SOFT,
        line_width=0,
    )
    timeline.add_vrect(
        x0=day_start + pd.Timedelta(hours=22),
        x1=day_start + pd.Timedelta(hours=24),
        fillcolor=ANOMALY_SOFT,
        line_width=0,
    )
    apply_layout(timeline, "Events of the day (shaded = night 22h-8h)")
    timeline.update_xaxes(title_text="Hour")
    timeline.update_yaxes(title_text="Sensor")

    hourly = diag["hourly_counts"]
    hist = go.Figure(
        go.Bar(x=list(hourly.index), y=hourly.values, marker_color=PRIMARY_SOFT)
    )
    apply_layout(hist, "Hourly distribution (basis of the entropy)")
    hist.update_xaxes(title_text="Hour of day")
    hist.update_yaxes(title_text="N events")

    return timeline, hist


def _plot_interval_statistics(
    group: pd.DataFrame, diag: dict
) -> tuple[go.Figure, go.Figure]:
    group = group.copy()
    group["timestamp"] = pd.to_datetime(group["timestamp"])
    ts_sorted = group.sort_values("timestamp")["timestamp"]

    timeline = go.Figure()
    timeline.add_trace(
        go.Scatter(
            x=ts_sorted,
            y=[0] * len(ts_sorted),
            mode="markers",
            marker={
                "size": 9,
                "color": PRIMARY_SOFT,
                "symbol": "line-ns",
                "line": {"width": 2, "color": PRIMARY_SOFT},
            },
            hoverinfo="skip",
        )
    )
    sample_pairs = list(zip(ts_sorted.iloc[:-1], ts_sorted.iloc[1:], strict=True))[:6]
    for t0, t1 in sample_pairs:
        timeline.add_shape(
            type="line",
            x0=t0,
            x1=t1,
            y0=0.15,
            y1=0.15,
            line={"color": ANOMALY, "width": 1.5, "dash": "dot"},
        )
    apply_layout(timeline, "Events of the day (red lines = illustrated intervals)")
    timeline.update_xaxes(title_text="Hour")
    timeline.update_yaxes(visible=False, range=[-1, 1])

    intervals_min = diag["intervals_seconds"] / 60.0
    hist = go.Figure(
        go.Histogram(x=intervals_min, nbinsx=25, marker_color=PRIMARY_SOFT)
    )
    apply_layout(hist, "Inter-event interval distribution (IEI)")
    hist.update_xaxes(title_text="Minutes between consecutive events")
    hist.update_yaxes(title_text="Frequency")

    return timeline, hist


def _plot_ngram_transition(diag: dict) -> tuple[go.Figure, go.Figure]:
    sequence = diag["sequence"]
    tokens = sequence[:40]

    strip = go.Figure()
    strip.add_trace(
        go.Scatter(
            x=list(range(len(tokens))),
            y=[0] * len(tokens),
            mode="markers+text",
            marker={"size": 14, "color": PRIMARY_SOFT},
            text=tokens,
            textposition="top center",
            textfont={"size": 9, "color": TEXT},
            hoverinfo="skip",
        )
    )
    for i in range(len(tokens) - 1):
        strip.add_annotation(
            x=i + 1,
            y=0,
            ax=i,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=ANOMALY,
        )
    apply_layout(strip, "Triggered sensor sequence (first 40 events)", height=260)
    strip.update_xaxes(visible=False)
    strip.update_yaxes(visible=False, range=[-1, 1])

    matrix = diag["transition_matrix"]
    heatmap = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues",
            showscale=False,
        )
    )
    apply_layout(heatmap, "Transition matrix (bigram sensor A -> sensor B)")

    return strip, heatmap


def _plot_next_event(
    group: pd.DataFrame, diag: dict
) -> tuple[go.Figure, go.Figure]:
    seq = diag["sequence"]
    tokens = seq[:40]
    transitions = diag["transitions"][:39]
    n = len(tokens)

    strip = go.Figure()
    strip.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[0] * n,
            mode="markers+text",
            marker={"size": 14, "color": PRIMARY_SOFT},
            text=tokens,
            textposition="top center",
            textfont={"size": 9, "color": TEXT},
            hoverinfo="skip",
        )
    )
    for i, tr in enumerate(transitions):
        rare = tr["rare"]
        strip.add_annotation(
            x=i + 1,
            y=0,
            ax=i,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=ANOMALY if rare else PRIMARY_SOFT,
        )
    apply_layout(strip, "Sensor sequence; red arrows = unlikely transitions", height=260)
    strip.update_xaxes(visible=False)
    strip.update_yaxes(visible=False, range=[-1, 1])

    logprobs = [tr["logprob"] for tr in diag["transitions"]]
    colors = [
        ANOMALY if tr["rare"] else PRIMARY_SOFT for tr in diag["transitions"]
    ]
    bar = go.Figure(
        go.Bar(
            x=list(range(len(logprobs))),
            y=logprobs,
            marker_color=colors,
            customdata=[tr["to"] for tr in diag["transitions"]],
            hovertemplate="→ %{customdata}<br>log P = %{y:.2f}<extra></extra>",
        )
    )
    apply_layout(bar, "Log-probability of each transition (lower = more anomalous)")
    bar.update_xaxes(title_text="Transition index")
    bar.update_yaxes(title_text="log P(next | current)")

    return strip, bar


def _load_stream(data_source: str) -> pd.DataFrame:
    """Build the event stream from the config already chosen in the Data step."""
    if data_source == "Synthetic":
        return generate_synthetic_events(
            n_days=int(st.session_state.get("fx_days", 4)),
            pattern=st.session_state.get("fx_pattern", "regular"),
            n_sensors=int(st.session_state.get("fx_sensors", 3)),
            events_per_day=int(st.session_state.get("fx_events_day", 80)),
            seed=42,
        )

    try:
        df = load_all_events()
    except Exception as e:  # noqa: BLE001 - DB may be missing until ingestion runs
        st.error(f"Could not load the database: {e}")
        st.stop()

    if df.empty:
        st.warning(
            "The database is empty. Run first `python src/ingestion/casas_loader.py`."
        )
        st.stop()

    max_days = int(st.session_state.get("fx_days_real", 10))
    timestamps = pd.to_datetime(df["timestamp"])
    dates = sorted(timestamps.dt.date.unique())[:max_days]
    return df[timestamps.dt.date.isin(dates)]


def _recap_stream(data_source: str) -> None:
    """One-line recap of the data currently inspected (configured in Data)."""
    if data_source == "Synthetic":
        st.caption(
            "Stream chosen in the **Data step**: pattern "
            f"*{st.session_state.get('fx_pattern', 'regular')}* · "
            f"{st.session_state.get('fx_days', 4)} days · "
            f"{st.session_state.get('fx_sensors', 3)} sensors · "
            f"{st.session_state.get('fx_events_day', 80)} events/day."
        )
    else:
        st.caption(
            f"Real data read from the database: first "
            f"{st.session_state.get('fx_days_real', 10)} days (configured in the Data step)."
        )


def _render_inspector(data_source: str, method_name: str) -> None:
    """Inspect a day: diagnostics charts + feature vectors.

    The event stream itself is configured in the Data step (no controls here).
    """
    df = _load_stream(data_source)
    _recap_stream(data_source)

    if method_name == "Window Aggregation":
        extractor = WindowAggregationExtractor()
    elif method_name == "Inter-Event Interval (IEI)":
        extractor = IntervalStatisticsExtractor()
    elif method_name == "Next-Event Prediction (Markov)":
        extractor = NextEventTransitionExtractor()
        extractor.fit(df)
    else:
        extractor = NGramTransitionExtractor()
        extractor.fit_vocabulary(df)

    X, dates = extractor.extract(df)
    df_dates = pd.to_datetime(df["timestamp"]).dt.date

    colored_section_header(
        "2",
        "🔍 Pick a Day to Inspect",
        METHOD_COLORS.get(method_name, FAMILY_BOUNDARY),
        "Raw events of each day → extracted numeric vector, feature by feature.",
    )
    selected_days = st.multiselect(
        "Day(s) to inspect in detail",
        list(dates),
        default=[next(iter(dates))],
        key="fx_day_pick",
    )
    chosen_days = selected_days or [next(iter(dates))]

    for day in chosen_days:
        st.markdown(f"### 📅 {day}")
        group = df[df_dates == day]
        diag = extractor.diagnostics(group)

        if method_name == "Window Aggregation":
            fig_a, fig_b = _plot_window_aggregation(group, diag)
        elif method_name == "Inter-Event Interval (IEI)":
            fig_a, fig_b = _plot_interval_statistics(group, diag)
        elif method_name == "Next-Event Prediction (Markov)":
            fig_a, fig_b = _plot_next_event(group, diag)
        else:
            fig_a, fig_b = _plot_ngram_transition(diag)

        chart_pair(fig_a, fig_b)
        col_legend_a, col_legend_b = st.columns(2)
        with col_legend_a:
            st.caption("Timeline: sensors on the Y axis, time on the X axis.")
        with col_legend_b:
            st.caption(
                "Distribution: intervals on the X axis, frequency on the Y axis."
            )
        if method_name == "N-gram Transition (Markov)":
            st.caption(
                "Strip: sensor sequence (X = event order). Matrix: P(next sensor | current)."
            )
        if method_name == "Next-Event Prediction (Markov)":
            st.caption(
                "Strip: sequence with arrows colored by likelihood. Bar: log P of each "
                "transition (low = anomalous)."
            )

    st.markdown("#### Feature vectors for the selected days")
    full_table = pd.DataFrame(X, columns=extractor.FEATURE_NAMES)
    full_table.insert(0, "date", dates)
    full_table = full_table[full_table["date"].isin(chosen_days)]
    st.dataframe(full_table, width="stretch", hide_index=True)
    st.caption(
        "One row per day: the extractor summarized the raw event stream into this numeric vector."
    )


def render_feature_extraction_view() -> None:
    breadcrumb(
        [("Data", False), ("Features", True), ("Detect", False), ("Ensemble", False)]
    )

    # Origin comes from the main Data step (CASAS track): Synthetic or Real.
    data_source = (
        "Real"
        if st.session_state.get("casas_source", "Synthetic") == "Real"
        else "Synthetic"
    )

    st.markdown("## How to turn events into numbers")
    st.caption(
        "Pick a method → see how it works → inspect a real day. "
        f"Data origin: **{data_source}** (chosen in the Data step)."
    )

    if "fx_method" not in st.session_state:
        st.session_state.fx_method = "Window Aggregation"

    current = st.session_state.fx_method
    colored_section_header(
        "1",
        "Extraction method",
        METHOD_COLORS.get(current, FAMILY_DISTANCE),
        "Each extractor looks at the event stream from a different angle.",
    )
    method_specs = [
        {
            "id": m,
            "icon": METHOD_GLYPHS[m],
            "title": m,
            "description": METHOD_CARDS[m]["desc"],
            "badge": METHOD_CARDS[m]["badge"],
            "color": METHOD_COLORS[m],
            "meta": [],
        }
        for m in METHOD_DESCRIPTIONS
    ]
    method = clickable_cards(method_specs, key="fx_method")

    display_chart(_method_schematic(method), key=f"fx_schematic_{method}")
    render_resources(method)

    st.markdown("---")
    _render_inspector(data_source, method)
