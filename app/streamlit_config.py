"""Centralized Streamlit UI configuration: detector registry and defaults.

Single source of truth for detector metadata (display name, description, category,
parameter ranges, and teaching-track overlay family). The sidebar sliders, the
teaching track and the results views all read from ``DETECTOR_REGISTRY``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectors.factory import DETECTOR_FACTORY


@dataclass(frozen=True)
class ParamSpec:
    """Widget definition for a single detector parameter.

    If ``options`` is provided the widget is a selectbox (enum param), otherwise a
    slider. The value is passed to the detector constructor as ``kwarg``.
    """

    label: str
    min: float
    max: float
    default: float
    step: float
    kwarg: str  # keyword passed to the detector constructor
    options: tuple[str, ...] = ()  # non-empty -> render a selectbox instead of a slider


@dataclass(frozen=True)
class DetectorSpec:
    """UI metadata for one detector."""

    name: str
    description: str
    category: str
    family: str = "none"  # teaching-track overlay: iforest / covariance_* / knn / lof / boundary_only / none
    default: bool = False
    params: tuple[ParamSpec, ...] = ()

    @property
    def detector_cls(self) -> type:
        return DETECTOR_FACTORY[self.name]


# ============================================================================
# DETECTOR REGISTRY (unified, used by sidebar + teaching track)
# ============================================================================

DETECTOR_REGISTRY: dict[str, DetectorSpec] = {
    "Isolation Forest": DetectorSpec(
        name="Isolation Forest",
        description=(
            "Random-partition isolation: anomalies are cut apart in a few splits. "
            "Fast and robust in high dimensions."
        ),
        category="Density",
        family="iforest",
        default=True,
        params=(
            ParamSpec("Contamination", 0.01, 0.20, 0.05, 0.01, "contamination"),
            ParamSpec("Trees", 20, 300, 100, 10, "n_estimators"),
            ParamSpec(
                "Max samples",
                0,
                0,
                "auto",
                0,
                "max_samples",
                options=("auto", "0.2", "0.4", "0.6", "0.8", "1.0"),
            ),
        ),
    ),
    "Extended IForest": DetectorSpec(
        name="Extended IForest",
        description="IForest with oblique (sliced) paths for ultra-high-dimensional data.",
        category="Density",
        family="iforest",
        params=(
            ParamSpec("Contamination", 0.01, 0.20, 0.05, 0.01, "contamination"),
            ParamSpec("Trees", 20, 300, 100, 10, "n_estimators"),
            ParamSpec(
                "Max samples",
                0,
                0,
                "auto",
                0,
                "max_samples",
                options=("auto", "0.2", "0.4", "0.6", "0.8", "1.0"),
            ),
        ),
    ),
    "Mahalanobis": DetectorSpec(
        name="Mahalanobis",
        description="Covariance-aware distance to the mean; assumes Gaussian data.",
        category="Distance",
        family="covariance_empirical",
        default=True,
        params=(
            ParamSpec("Threshold percentile", 80, 99, 95, 1, "threshold_percentile"),
        ),
    ),
    "Elliptic Envelope": DetectorSpec(
        name="Elliptic Envelope",
        description="Robust Gaussian fit (MCD) resistant to outliers during training.",
        category="Distance",
        family="covariance_elliptic",
        default=False,
        params=(ParamSpec("Contamination", 0.01, 0.30, 0.10, 0.01, "contamination"),),
    ),
    "Robust Covariance": DetectorSpec(
        name="Robust Covariance",
        description="Minimum Covariance Determinant ellipse; very robust covariance estimate.",
        category="Distance",
        family="covariance_robust",
        params=(ParamSpec("Contamination", 0.01, 0.30, 0.10, 0.01, "contamination"),),
    ),
    "KNN": DetectorSpec(
        name="KNN",
        description="Distance to the k-th nearest neighbor; distribution-agnostic.",
        category="Distance",
        family="knn",
        params=(
            ParamSpec("Neighbors", 3, 50, 5, 1, "n_neighbors"),
            ParamSpec("Contamination", 0.01, 0.30, 0.10, 0.01, "contamination"),
        ),
    ),
    "OC-SVM": DetectorSpec(
        name="OC-SVM",
        description="Learns a boundary that wraps the normal region; supports kernels.",
        category="Boundary",
        family="boundary_only",
        params=(
            ParamSpec("Nu", 0.01, 0.30, 0.05, 0.01, "nu"),
            ParamSpec(
                "Gamma",
                0,
                0,
                "auto",
                0,
                "gamma",
                options=("auto", "scale", "0.01", "0.1", "0.5", "1.0"),
            ),
            ParamSpec(
                "Kernel",
                0,
                0,
                "rbf",
                0,
                "kernel",
                options=("rbf", "linear", "poly"),
            ),
        ),
    ),
    "LOF": DetectorSpec(
        name="LOF",
        description="Local density factor; compares local vs. global density.",
        category="Density",
        family="lof",
        params=(
            ParamSpec("Neighbors", 5, 50, 20, 1, "n_neighbors"),
            ParamSpec("Contamination", 0.01, 0.30, 0.05, 0.01, "contamination"),
        ),
    ),
    "Z-Score": DetectorSpec(
        name="Z-Score",
        description="Per-feature standard-deviation score; flags points far from the training mean.",
        category="Univariate",
        family="boundary_only",
        params=(ParamSpec("Threshold (std)", 1.0, 5.0, 3.0, 0.1, "threshold"),),
    ),
    "PCA Reconstruction": DetectorSpec(
        name="PCA Reconstruction",
        description="Reconstruction error to the dominant linear subspace.",
        category="Dimensionality",
        family="boundary_only",
        params=(
            ParamSpec("Components", 1, 20, 5, 1, "n_components"),
            ParamSpec("Threshold percentile", 50, 99, 95, 1, "threshold_percentile"),
        ),
    ),
    "HMM": DetectorSpec(
        name="HMM",
        description="Hidden Markov chain; regime transitions in temporal data.",
        category="Sequential",
        params=(ParamSpec("States", 2, 8, 3, 1, "n_components"),),
    ),
    "Hawkes": DetectorSpec(
        name="Hawkes",
        description="Self-exciting point process for event streams.",
        category="Sequential",
        params=(ParamSpec("Decay", 0.1, 1.0, 0.5, 0.1, "decay"),),
    ),
}

# Display order = insertion order of the registry.
DETECTOR_NAMES: list[str] = list(DETECTOR_REGISTRY.keys())

DETECTOR_DEFAULTS_LIST: list[str] = [
    name for name, spec in DETECTOR_REGISTRY.items() if spec.default
]

# Guided-mode preset: one detector per major family (density, covariance, local
# density, sequential). Advanced mode lets the user pick any of the 12.
SIMPLE_MODE_DETECTORS: list[str] = [
    "Isolation Forest",
    "Mahalanobis",
    "LOF",
    "HMM",
]

SIMPLE_MODE_EXPLANATION = (
    "One detector per family: **Isolation Forest** (density), **Mahalanobis** "
    "(covariance), **LOF** (local density) and **HMM** (temporal patterns). "
    "Defaults are fine — switch to *Advanced* to tune everything."
)

DETECTOR_CATEGORIES: dict[str, list[str]] = {}
for spec in DETECTOR_REGISTRY.values():
    DETECTOR_CATEGORIES.setdefault(spec.category, []).append(spec.name)

# ============================================================================
# PIPELINE DEFAULTS
# ============================================================================

ENSEMBLE_DEFAULTS = {
    "mode": "soft",
    "threshold_percentile": 90,
    "weighting_scheme": "Uniform",
}

TRAINING_DEFAULTS = {
    "train_split": 0.7,
}

SYNTHETIC_EVALUATION_DEFAULTS = {
    "contamination": 0.15,
    "magnitude": 6.0,
}
