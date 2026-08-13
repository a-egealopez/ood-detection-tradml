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
from evaluation.synthetic_injection import (
    evaluate_with_synthetic_anomalies,
)
from features import FeatureScaler, TemporalFeatureExtractor

EPSILON = 1e-10
MIN_DAYS = 10
MIN_HOLDOUT = 5


@dataclass
class HouseResult:
    """Container with everything a view needs to render one house's outcome."""

    df: pd.DataFrame
    details: pd.DataFrame
    anomalies: np.ndarray
    scores: np.ndarray
    ensemble: EnsembleDetector
    scaler: FeatureScaler
    n_holdout: int
    synthetic_metrics: dict | None = None
    X_scaled: np.ndarray | None = None
    X_holdout: np.ndarray | None = None


def compute_weights(
    detectors: list, X_train: np.ndarray, scheme: str = "Uniform"
) -> np.ndarray:
    """Weight per detector: uniform or inverse-entropy (confidence-based)."""
    n_detectors = len(detectors)
    if n_detectors == 0:
        return np.array([])

    if scheme == "Uniform":
        return np.array([1.0 / n_detectors] * n_detectors)

    if scheme == "Entropy-based":
        weights = []
        for detector in detectors:
            detector.fit(X_train)
            _, scores_train = detector.predict(X_train)
            entropy = -np.mean(
                scores_train * np.log(scores_train + EPSILON)
                + (1 - scores_train) * np.log(1 - scores_train + EPSILON)
            )
            confidence = 1.0 / (1.0 + entropy)
            weights.append(confidence)
        weights = np.array(weights)

        # Degenerate cases (NaN scores from singular covariance, etc.) fall back
        # to uniform weighting instead of crashing the ensemble.
        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            return np.ones(n_detectors) / n_detectors
        return weights / weights.sum()

    raise ValueError(f"Unknown weighting scheme: {scheme!r}")


def run_house(
    house_id: str,
    df: pd.DataFrame,
    *,
    detector_names: list[str],
    detector_params: dict,
    train_split: float = 0.7,
    weighting_scheme: str = "Uniform",
    ensemble_mode: str = "soft",
    threshold_percentile: float = 90,
    contamination: float = 0.15,
    magnitude: float = 6.0,
) -> tuple[HouseResult | None, str | None]:
    """Run the full pipeline on one house. Returns ``(result, error)``.

    ``df`` must already be loaded (the app caches it); this function stays
    responsible for feature extraction, scaling, training and evaluation.
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
    X_train, X_holdout = X_scaled[:split_idx], X_scaled[split_idx:]

    detectors = build_detectors(detector_names, detector_params)
    if not detectors:
        return None, "Select at least one detector."

    weights = compute_weights(detectors, X_train, weighting_scheme)

    ensemble = EnsembleDetector(
        detectors=detectors,
        weights=weights,
        ensemble_mode=ensemble_mode,
        ensemble_threshold_percentile=threshold_percentile,
    )
    ensemble.fit(X_train)

    anomalies, scores, details = ensemble.predict(X_scaled)
    details["date"] = dates

    synthetic_metrics = None
    if len(X_holdout) >= MIN_HOLDOUT:
        synthetic_metrics = evaluate_with_synthetic_anomalies(
            ensemble,
            X_holdout,
            contamination=contamination,
            magnitude=magnitude,
        )

    result = HouseResult(
        df=df,
        details=details,
        anomalies=anomalies,
        scores=scores,
        ensemble=ensemble,
        scaler=scaler,
        n_holdout=len(X_holdout),
        synthetic_metrics=synthetic_metrics,
        X_scaled=X_scaled,
        X_holdout=X_holdout,
    )
    return result, None
