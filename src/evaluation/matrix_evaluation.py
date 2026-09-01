"""Matrix evaluation: anomaly type x intensity x detector x seed.

The full-coherence evaluation demanded by the plan (Fase 4). Features are
extracted from the *raw event stream* (so contextual/collective anomalies are
injected where the sequential detectors live), scaled on the training split only
(no look-ahead), and each detector scores the full train+eval sequence, taking
the eval tail (so Hawkes state and HMM conditioning carry across the
train->holdout boundary; vectorial detectors are per-row and unaffected).

Anomaly types
-------------
- ``point``: event-level night burst (``inject_point_events``) — a loud day.
- ``contextual``: whole-day circular time shift of the routine (``event_injection``).
- ``collective``: event-level intra-day reordering (``event_injection``).
- ``control``: null control — clean holdout with random labels; AUROC must be ~0.5.

All three anomaly types are injected on the *raw event stream* at intensity
low/medium/high and the features are re-extracted afterwards (there is no
feature-level injection anywhere in the evaluation).

Cost note: the shared inputs (feature views, transition model, injected eval
matrices) are prepared once per (house, type, intensity, seed); only the per
-detector fit/predict is repeated inside the seed loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detectors import (
    HawkesDetector,
    HMMDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    MarkovSequenceDetector,
    PCAReconstructionDetector,
    ZScoreDetector,
)
from detectors.base import BaseDetector
from detectors.constants import (
    DEFAULT_CONTAMINATION,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_SPLIT,
)
from evaluation.event_injection import (
    COLLECTIVE_INTENSITIES,
    CONTEXTUAL_INTENSITIES,
    POINT_INTENSITIES,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    select_anomaly_dates,
)
from evaluation.metrics import compute_metrics
from features import FeatureScaler, NextEventTransitionExtractor, TemporalFeatureExtractor

# One detector per family the success criteria care about. The builders receive
# the seed so stochastic detectors get a fresh draw per seed.
DEFAULT_DETECTORS: dict[str, Callable[[int], BaseDetector]] = {
    "Z-Score": lambda _: ZScoreDetector(),
    "Mahalanobis": lambda _: MahalanobisDetector(),
    "Isolation Forest": lambda seed: IsolationForestDetector(random_state=seed),
    "PCA Reconstruction": lambda _: PCAReconstructionDetector(),
    "HMM": lambda seed: HMMDetector(random_state=seed),
    "Hawkes": lambda _: HawkesDetector(),
    "Markov Sequence": lambda _: MarkovSequenceDetector(),
}


@dataclass
class CellResult:
    house: str
    anomaly_type: str
    intensity: str
    detector: str
    seed: int
    auroc: float
    precision: float
    recall: float
    f1: float


@dataclass
class CleanViews:
    df: pd.DataFrame
    dates: list
    train_mask: np.ndarray
    X9_scaled: np.ndarray
    X_hourly: np.ndarray
    X_trans: np.ndarray
    trans_extractor: NextEventTransitionExtractor
    scaler: FeatureScaler


@dataclass
class PreparedCell:
    """Everything a detector cell needs after injection/re-extraction.

    The transition features (``X_eval_trans``) are computed lazily by
    ``evaluate_detector`` for the Markov Sequence detector only; every other
    detector is blind to the transition view, so the expensive re-extraction is
    skipped unless actually needed.
    """

    anomaly_type: str
    intensity: str
    seed: int
    y_holdout: np.ndarray
    holdout_pos: np.ndarray
    X_eval9: np.ndarray
    X_eval_hourly: np.ndarray
    df_inj: pd.DataFrame
    df_train_events: pd.DataFrame


def prepare_house(df_house, train_split: float) -> CleanViews:
    """Normalize a house and build the shared clean views (once per house)."""
    df = df_house.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    dates = sorted(df["date"].unique())

    extractor = TemporalFeatureExtractor()
    X9, _ = extractor.extract(df)
    X_hourly = extractor.hourly_counts(df)

    split_idx = max(1, int(len(dates) * train_split))
    train_dates = set(dates[:split_idx])
    train_mask = np.array([d in train_dates for d in dates])

    scaler = FeatureScaler()
    scaler.fit(X9[train_mask])
    X9_scaled = scaler.transform(X9)

    # Transition model: fitted on the clean training events only; per-day features
    # extracted on the clean full stream (re-extracted per cell if injected).
    trans_extractor = NextEventTransitionExtractor()
    trans_extractor.fit(df[df["date"].isin(train_dates)])
    X_trans = trans_extractor.extract(df)[0]

    return CleanViews(
        df=df,
        dates=dates,
        train_mask=train_mask,
        X9_scaled=X9_scaled,
        X_hourly=X_hourly,
        X_trans=X_trans,
        trans_extractor=trans_extractor,
        scaler=scaler,
    )


def prepare_cell(
    views: CleanViews,
    _house_id: str,
    anomaly_type: str,
    intensity: str,
    seed: int,
    *,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    contamination: float = DEFAULT_CONTAMINATION,
    intensity_fractions: dict | None = None,
) -> PreparedCell:
    """Inject (if needed) and re-extract the eval matrices for one seed."""
    rng = np.random.default_rng(seed)
    df, dates, train_mask = views.df, views.dates, views.train_mask

    split_idx = max(1, int(len(dates) * train_split))
    train_dates = set(dates[:split_idx])
    holdout_dates = dates[split_idx:]
    n_holdout = len(holdout_dates)
    if n_holdout < 5:
        raise ValueError("Holdout too small for a stable AUROC")
    n_anomaly = max(1, round(n_holdout * contamination))

    holdout_pos = np.where(~train_mask)[0]

    if anomaly_type == "control":
        # Null control: clean holdout, random labels -> AUROC must be ~0.5.
        X_eval9, X_eval_hourly = views.X9_scaled, views.X_hourly
        df_inj = df
        anomaly_dates = set(rng.choice(holdout_dates, size=n_anomaly, replace=False))
    else:
        # Event-level injection (point / contextual / collective), then features
        # are re-extracted from the injected stream and scaled with the train-only
        # scaler. This is the only injection path in the whole evaluation.
        fraction = (intensity_fractions or {}).get(anomaly_type, {}).get(intensity, 0.5)
        anomaly_dates = select_anomaly_dates(holdout_dates, rng, n_anomaly)
        if anomaly_type == "point":
            df_inj = inject_point_events(df, rng, fraction, anomaly_dates)
        elif anomaly_type == "contextual":
            df_inj = inject_contextual_events(df, rng, fraction, anomaly_dates)
        elif anomaly_type == "collective":
            df_inj = inject_collective_events(df, rng, fraction, anomaly_dates)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        X9_inj, _ = TemporalFeatureExtractor().extract(df_inj)
        X_eval9 = views.scaler.transform(X9_inj)
        X_eval_hourly = TemporalFeatureExtractor().hourly_counts(df_inj)

    y = np.zeros(len(dates), dtype=int)
    for day_offset, d in enumerate(dates):
        if d in anomaly_dates:
            y[day_offset] = 1

    return PreparedCell(
        anomaly_type=anomaly_type,
        intensity=intensity,
        seed=seed,
        y_holdout=y[holdout_pos],
        holdout_pos=holdout_pos,
        X_eval9=X_eval9,
        X_eval_hourly=X_eval_hourly,
        df_inj=df_inj,
        df_train_events=cast(pd.DataFrame, df[df["date"].isin(train_dates)]),
    )


def evaluate_detector(
    cell: PreparedCell,
    views: CleanViews,
    house_id: str,
    detector: str,
    seed: int,
) -> CellResult:
    """Fit one detector on the training prefix and score the eval tail."""
    builder = DEFAULT_DETECTORS[detector]
    train_mask = views.train_mask
    if detector == "Markov Sequence":
        det = cast(MarkovSequenceDetector, builder(seed))
        # Reuse the house-level transition extractor (already fitted on the clean
        # training events) instead of refitting it for every cell.
        det.extractor = views.trans_extractor
        det.fit(views.X_trans[train_mask])
        X_eval_trans = views.trans_extractor.extract(cell.df_inj)[0]
        _, scores = det.predict(X_eval_trans)
    elif detector == "Hawkes":
        det = builder(seed)
        det.fit(views.X_hourly[train_mask])
        _, scores = det.predict(cell.X_eval_hourly)
    else:
        det = builder(seed)
        det.fit(views.X9_scaled[train_mask])
        _, scores = det.predict(cell.X_eval9)

    scores_holdout = scores[cell.holdout_pos]
    y_holdout = cell.y_holdout
    auroc = float(roc_auc_score(y_holdout, scores_holdout))
    y_pred = (scores_holdout > np.median(scores_holdout)).astype(int)
    metrics = compute_metrics(y_holdout, y_pred)

    return CellResult(
        house=house_id,
        anomaly_type=cell.anomaly_type,
        intensity=cell.intensity,
        detector=detector,
        seed=seed,
        auroc=auroc,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
    )


def run_matrix(
    df_house,
    house_id: str,
    *,
    anomaly_types=("point", "contextual", "collective", "control"),
    intensities=("low", "medium", "high"),
    detectors=None,
    n_seeds: int = 10,
    seed_base: int = DEFAULT_RANDOM_STATE,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    contamination: float = DEFAULT_CONTAMINATION,
) -> pd.DataFrame:
    """Full matrix as one row per (type, intensity, detector, seed)."""
    detectors = detectors or list(DEFAULT_DETECTORS.keys())
    intensity_fractions = {
        "point": POINT_INTENSITIES,
        "contextual": CONTEXTUAL_INTENSITIES,
        "collective": COLLECTIVE_INTENSITIES,
    }

    views = prepare_house(df_house, train_split)

    rows = []
    for anomaly_type in anomaly_types:
        levels = (
            intensities
            if anomaly_type in ("point", "contextual", "collective")
            else ("none",)
        )
        for level in levels:
            for seed in range(seed_base, seed_base + n_seeds):
                cell = prepare_cell(
                    views,
                    house_id,
                    anomaly_type,
                    level,
                    seed,
                    train_split=train_split,
                    contamination=contamination,
                    intensity_fractions=intensity_fractions,
                )
                for detector in detectors:
                    rows.append(evaluate_detector(cell, views, house_id, detector, seed))
    return pd.DataFrame([r.__dict__ for r in rows])


def aggregate_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-seed matrix into mean +/- std per (type, intensity, detector)."""
    groups = matrix.groupby(["anomaly_type", "intensity", "detector"])
    agg = groups.agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg


def monotonicity_check(agg: pd.DataFrame, expected_winners: dict) -> dict:
    """Check that expected winners' AUROC rises with intensity.

    ``expected_winners`` maps anomaly_type -> list of detector names. Only
    intensity-graded types (point/contextual/collective) are checked; the null
    control is single-level and skipped. Returns a dict {detector: (ok,
    auroc_by_intensity)} with AUROC ordered low -> medium -> high.
    """
    LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
    results = {}
    for anomaly_type, detectors in expected_winners.items():
        for detector in detectors:
            cells = agg[
                (agg["anomaly_type"] == anomaly_type)
                & (agg["detector"] == detector)
                & (agg["intensity"].isin(LEVEL_ORDER))
            ]
            if cells.empty:
                results[f"{anomaly_type}/{detector}"] = (None, [])
                continue
            cells = cells.copy()
            cells["level_order"] = cells["intensity"].map(LEVEL_ORDER)
            cells = cells.sort_values("level_order")
            ok = bool(cells["auroc_mean"].is_monotonic_increasing)
            results[f"{anomaly_type}/{detector}"] = (
                ok,
                cells["auroc_mean"].round(3).tolist(),
            )
    return results
