import numpy as np
import pandas as pd

from detectors.sequential.markov_sequence_detector import MarkovSequenceDetector
from ingestion.casas_stream_generator import build_weighted_graph, simulate_day


def _stream_of_day(rng, graph, day_offset: int, reverse: bool = False) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    rows = simulate_day(
        rng,
        start + pd.Timedelta(days=day_offset),
        180,
        0.08,
        graph,
        {"name": "typical", "event_factor": 1.0, "night_offset": 0.0},
    )
    if reverse:
        rows = rows[::-1]
    df = pd.DataFrame(rows, columns=["date", "time", "sensor_id", "reading"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("timestamp")
    return df[["timestamp", "sensor_id"]]


def test_reversed_day_scores_higher():
    """Acceptance: a reversed day (rare transitions) must score higher than a typical one."""
    rng = np.random.default_rng(7)
    graph = build_weighted_graph({"door_activity": 1.0})

    train = pd.concat([_stream_of_day(rng, graph, d) for d in range(30)])
    typical = _stream_of_day(rng, graph, 31)
    rare = _stream_of_day(rng, graph, 32, reverse=True)

    detector = MarkovSequenceDetector()
    detector.fit_extractor(train)
    detector.fit(detector.extractor.extract(train)[0])

    _, score_typical = detector.predict(detector.extractor.extract(typical)[0])
    _, score_rare = detector.predict(detector.extractor.extract(rare)[0])

    assert float(score_rare[0]) > float(score_typical[0])
    detector._assert_unit_range(score_rare)
