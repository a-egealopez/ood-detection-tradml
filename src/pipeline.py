"""CASAS anomaly-detection pipeline (pure logic, no Streamlit imports).

Extracts daily features, scales them, trains an ensemble of detectors and returns
per-house results. Kept outside ``app/`` so it stays importable from the CLI, tests,
and the Streamlit UI alike.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors import EnsembleDetector
from detectors.factory import build_detectors
from features import FeatureScaler, TemporalFeatureExtractor

MIN_DAYS = 10


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
    train_split: float = 0.7,
    ensemble_mode: str = "soft",
    threshold_percentile: float = 90,
) -> tuple[HouseResult | None, str | None]:
    """Run the full pipeline on one house. Returns ``(result, error)``.

    ``df`` must already be loaded (the app caches it); this function stays
    responsible for feature extraction, scaling and training.
    """
    if df is None or df.empty:
        return None, f"No data for '{house_id}'."

    extractor = TemporalFeatureExtractor()
    X, dates = extractor.extract(df)

    if len(X) < MIN_DAYS:
        return None, f"Too few days in '{house_id}' ({len(X)})."

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = max(1, int(len(X_scaled) * train_split))
    X_train = X_scaled[:split_idx]

    detectors = build_detectors(detector_names, detector_params)
    if not detectors:
        return None, "Select at least one detector."

    weights = np.ones(len(detectors)) / len(detectors)

    ensemble = EnsembleDetector(
        detectors=detectors,
        weights=weights,
        ensemble_mode=ensemble_mode,
        ensemble_threshold_percentile=threshold_percentile,
    )
    ensemble.fit(X_train)

    anomalies, scores, details = ensemble.predict(X_scaled)
    details["date"] = dates

    return HouseResult(details=details, anomalies=anomalies, scores=scores, ensemble=ensemble), None
