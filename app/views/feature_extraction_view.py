"""Feature extraction tutorial: from an event stream to a feature vector.

Lets the user choose among three event-driven time-series methods and inspect, for a
selected day, the step-by-step computation behind the extracted vector.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components import chart_pair  # noqa: E402
from data_access import load_all_events  # noqa: E402
from features.event_driven_extractors import (  # noqa: E402
    IntervalStatisticsExtractor,
    NGramTransitionExtractor,
    WindowAggregationExtractor,
    generate_synthetic_events,
)
from theme import ANOMALY, ANOMALY_SOFT, PRIMARY_SOFT, TEXT, apply_layout  # noqa: E402

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
}

METHOD_ANOMALY_TYPES = {
    "Window Aggregation": (
        "Contextual + Collective (full day)",
        (
            "The context (hour/time-slot) is encoded in the features (e.g. "
            "`night_activity_ratio`) and each vector summarizes a whole day: it detects "
            "atypical days as a set, not isolated events."
        ),
    ),
    "Inter-Event Interval (IEI)": (
        "Point (raw) or Collective (aggregated)",
        (
            "Raw intervals can flag a single anomalous gap (point). The vector extracted "
            "here is aggregated per day, so it detects days with an atypical overall "
            "'rhythm' (collective), not the exact instant of the gap."
        ),
    ),
    "N-gram Transition (Markov)": (
        "Collective sequence (pattern-based)",
        (
            "Only looks at the order of events, not their instant or magnitude: it "
            "detects routines that break the usual sequence pattern even when each "
            "sensor and interval is individually normal."
        ),
    ),
}


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
        go.Bar(
            x=list(hourly.index), y=hourly.values, marker_color=PRIMARY_SOFT
        )
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
    sample_pairs = list(zip(ts_sorted.iloc[:-1], ts_sorted.iloc[1:]))[:6]
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


def _load_synthetic_events() -> pd.DataFrame:
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        pattern = st.selectbox("Temporal pattern", ["regular", "bursty", "day_night"])
    with s2:
        n_days = st.slider("Days", 2, 10, 4)
    with s3:
        n_sensors = st.slider("Sensors", 2, 6, 3)
    with s4:
        events_per_day = st.slider("Events/day", 20, 200, 80, step=10)

    return generate_synthetic_events(
        n_days=n_days,
        pattern=pattern,
        n_sensors=n_sensors,
        events_per_day=events_per_day,
        seed=42,
    )


def _load_real_events() -> pd.DataFrame:
    try:
        df = load_all_events()
    except Exception as e:
        st.error(f"Could not load the database: {e}")
        st.stop()

    if df.empty:
        st.warning(
            "The database is empty. Run first `python src/ingestion/casas_loader.py`."
        )
        st.stop()

    max_days = st.slider(
        "Days to analyze (from the start of the dataset)", 3, 30, 10
    )
    timestamps = pd.to_datetime(df["timestamp"])
    dates = sorted(timestamps.dt.date.unique())[:max_days]
    return df[timestamps.dt.date.isin(dates)]


def render_feature_extraction_view() -> None:
    st.markdown(
        """
        ## Feature Extraction: From Event Stream to Feature Vector

        Choose the data origin and the extraction method to see, step by step, how a set
        of raw events (timestamp, sensor, type) becomes the numeric vector consumed by
        the anomaly detectors.
        """
    )

    col_source, col_method = st.columns([1, 1.4])
    with col_source:
        data_source = st.radio(
            "Data origin", ["Synthetic", "Real (CASAS Aruba)"], horizontal=True
        )
    with col_method:
        method_name = st.radio(
            "Extraction method", list(METHOD_DESCRIPTIONS.keys()), horizontal=True
        )

    st.caption(METHOD_DESCRIPTIONS[method_name])
    anomaly_type, anomaly_rationale = METHOD_ANOMALY_TYPES[method_name]
    st.info(f"**Anomaly type detected: {anomaly_type}.** {anomaly_rationale}")

    if data_source == "Synthetic":
        df = _load_synthetic_events()
    else:
        df = _load_real_events()

    st.markdown("---")

    if method_name == "Window Aggregation":
        extractor = WindowAggregationExtractor()
    elif method_name == "Inter-Event Interval (IEI)":
        extractor = IntervalStatisticsExtractor()
    else:
        extractor = NGramTransitionExtractor()
        extractor.fit_vocabulary(df)

    X, dates = extractor.extract(df)
    df_dates = pd.to_datetime(df["timestamp"]).dt.date

    selected_day = st.selectbox("Day to inspect in detail", list(dates))
    group = df[df_dates == selected_day]

    diag = extractor.diagnostics(group)

    if method_name == "Window Aggregation":
        fig_a, fig_b = _plot_window_aggregation(group, diag)
    elif method_name == "Inter-Event Interval (IEI)":
        fig_a, fig_b = _plot_interval_statistics(group, diag)
    else:
        fig_a, fig_b = _plot_ngram_transition(diag)

    chart_pair(fig_a, fig_b)

    st.markdown(f"#### Extracted feature vector - {selected_day}")
    feature_row = pd.DataFrame([diag["features"]], columns=diag["feature_names"])
    st.dataframe(feature_row, use_container_width=True, hide_index=True)

    st.markdown("#### Extracted vectors for every day in the dataset")
    full_table = pd.DataFrame(X, columns=extractor.FEATURE_NAMES)
    full_table.insert(0, "date", dates)
    st.dataframe(full_table, use_container_width=True, hide_index=True)
