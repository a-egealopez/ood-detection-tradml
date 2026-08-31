"""Detector factory: maps display names to detector classes and builds instances.

Kept outside Streamlit so it stays importable from the CLI, the app, and tests.
Parameter values are cast to int when the default is an int (sliders return floats);
numeric-looking strings (e.g. "0.5") become floats.
"""

import contextlib
from typing import Any

from .base import BaseDetector
from .sequential.hawkes_detector import HawkesDetector
from .sequential.hmm_detector import HMMDetector
from .sequential.markov_sequence_detector import MarkovSequenceDetector
from .vectorial import (
    EllipticEnvelopeDetector,
    IsolationForestDetector,
    KNNDetector,
    LOFDetector,
    MahalanobisDetector,
    OCSVMDetector,
    PCAReconstructionDetector,
    RobustCovarianceDetector,
    ZScoreDetector,
)

_DETECTOR_MAP: dict[str, tuple[type[BaseDetector], dict]] = {
    "Isolation Forest": (IsolationForestDetector, {}),
    "Extended IForest": (IsolationForestDetector, {"sliced_path": True}),
    "Mahalanobis": (MahalanobisDetector, {}),
    "Elliptic Envelope": (EllipticEnvelopeDetector, {}),
    "Robust Covariance": (RobustCovarianceDetector, {}),
    "KNN": (KNNDetector, {}),
    "OC-SVM (RBF)": (OCSVMDetector, {"kernel": "rbf"}),
    "OC-SVM (Linear)": (OCSVMDetector, {"kernel": "linear"}),
    "OC-SVM (Poly)": (OCSVMDetector, {"kernel": "poly"}),
    "LOF": (LOFDetector, {}),
    "Z-Score": (ZScoreDetector, {}),
    "PCA Reconstruction": (PCAReconstructionDetector, {}),
    "HMM": (HMMDetector, {}),
    "Hawkes": (HawkesDetector, {}),
    "Markov Sequence": (MarkovSequenceDetector, {}),
}

# Derived: name -> class (for backward compat / inspection)
DETECTOR_FACTORY: dict[str, type[BaseDetector]] = {k: v[0] for k, v in _DETECTOR_MAP.items()}


def _coerce_param(value: Any) -> Any:
    """Cast float integers to int; parse numeric strings to float."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return value


def build_detector(name: str, params: dict) -> BaseDetector:
    """Instantiate a detector by display name with the given parameter kwargs."""
    if name not in _DETECTOR_MAP:
        raise ValueError(f"Unknown detector: {name!r}")

    detector_cls, fixed = _DETECTOR_MAP[name]
    kwargs = {k: _coerce_param(v) for k, v in params.items()}
    kwargs.update(fixed)
    return detector_cls(**kwargs)


def build_detectors(names: list[str], params_by_detector: dict) -> list[BaseDetector]:
    """Build one detector per display name, passing each its configured params."""
    return [build_detector(name, params_by_detector.get(name, {})) for name in names]
