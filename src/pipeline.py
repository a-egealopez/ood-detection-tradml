import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_DAYS
from detectors import EnsembleDetector
from detectors.constants import (
    DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
    DEFAULT_TRAIN_SPLIT,
)
from detectors.factory import build_detectors
from detectors.sequential.hawkes_detector import HawkesDetector
from detectors.sequential.markov_sequence_detector import MarkovSequenceDetector
from features import FeatureScaler, TemporalFeatureExtractor


@dataclass
class HouseResult:
    """Container with everything a view needs to render one house's outcome."""

    details: pd.DataFrame
    anomalies: np.ndarray
    scores: np.ndarray
    ensemble: EnsembleDetector


def run_house(
    house_id: str,
    df: pd.DataFrame,
    *,
    detector_names: list[str],
    detector_params: dict,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    ensemble_mode: Literal["soft", "hard"] = "soft",
    threshold_percentile: float = DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
) -> tuple[HouseResult | None, str | None]:
    """Run the full pipeline on one house. Returns ``(result, error)``.

    ``df`` must already be loaded (the app caches it); this function stays
    responsible for feature extraction, scaling and training.
    """
    if df.empty:
        return None, f"No data for '{house_id}'."

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df)

    if len(X) < MIN_DAYS:
        return None, f"Too few days in '{house_id}' ({len(X)})."

    # Raw per-hour daily counts (unscaled). Only the Hawkes detector consumes
    # these: its Poisson intensity model is valid for counts, not for the
    # z-scored continuous features every other detector receives. Hourly counts
    # (instead of the 3 daily aggregates) make it sensitive to *when* during the
    # day events happen (contextual anomalies) while staying blind to intra-day
    # order (collective anomalies).
    X_hourly = extractor.hourly_counts(df)

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = max(1, int(len(X_scaled) * train_split))
    X_train = X_scaled[:split_idx]

    detectors = build_detectors(detector_names, detector_params)
    if not detectors:
        return None, "Select at least one detector."

    # MarkovSequence detectors learn their transition probabilities from the
    # training events only (no look-ahead) and consume the per-day transition
    # feature matrix extracted with that model.
    train_dates = set(dates[:split_idx])
    df_train_events = df[df["date"].isin(train_dates)]
    X_transition = None
    for detector in detectors:
        if isinstance(detector, MarkovSequenceDetector):
            detector.fit_extractor(df_train_events)
            X_transition = detector.extractor.extract(df)[0]
            break

    weights = np.ones(len(detectors)) / len(detectors)

    ensemble = EnsembleDetector(
        detectors=detectors,
        weights=weights,
        ensemble_mode=ensemble_mode,
        ensemble_threshold_percentile=threshold_percentile,
        detector_inputs=[
            X_hourly
            if isinstance(detector, HawkesDetector)
            else X_transition
            if isinstance(detector, MarkovSequenceDetector)
            else None
            for detector in detectors
        ],
    )
    ensemble.fit(X_train)

    anomalies, scores, details = ensemble.predict(X_scaled)
    details["date"] = dates

    return HouseResult(details=details, anomalies=anomalies, scores=scores, ensemble=ensemble), None
