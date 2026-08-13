"""Detector factory: maps display names to detector classes and builds instances.

Kept outside Streamlit so it stays importable from the CLI, the app, and tests.
Parameter values are cast to int when the default is an int (sliders return floats).
"""

from .sequential.hawkes_detector import HawkesDetector
from .sequential.hmm_detector import HMMDetector
from .vectorial import (
    EllipticEnvelopeDetector,
    ExtendedIForestDetector,
    IsolationForestDetector,
    KNNDetector,
    LOFDetector,
    MahalanobisDetector,
    OCSVMDetector,
    PCAReconstructionDetector,
    RobustCovarianceDetector,
    ZScoreDetector,
)

DETECTOR_FACTORY: dict[str, type] = {
    "Isolation Forest": IsolationForestDetector,
    "Extended IForest": ExtendedIForestDetector,
    "Mahalanobis": MahalanobisDetector,
    "Elliptic Envelope": EllipticEnvelopeDetector,
    "Robust Covariance": RobustCovarianceDetector,
    "KNN": KNNDetector,
    "OC-SVM": OCSVMDetector,
    "LOF": LOFDetector,
    "Z-Score": ZScoreDetector,
    "PCA Reconstruction": PCAReconstructionDetector,
    "HMM": HMMDetector,
    "Hawkes": HawkesDetector,
}


def build_detector(name: str, params: dict) -> object:
    """Instantiate a detector by display name with the given parameter kwargs."""
    if name not in DETECTOR_FACTORY:
        raise ValueError(f"Unknown detector: {name!r}")

    detector_cls = DETECTOR_FACTORY[name]
    kwargs = {}
    for key, value in params.items():
        # Sliders return floats; integer parameters need casting back to int.
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        # Selectbox string params that encode a number (e.g. max_samples="0.5")
        # become real floats so sklearn receives the numeric value.
        elif isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        kwargs[key] = value
    return detector_cls(**kwargs)


def build_detectors(names: list[str], params_by_detector: dict) -> list:
    """Build one detector per display name, passing each its configured params."""
    return [build_detector(name, params_by_detector.get(name, {})) for name in names]
