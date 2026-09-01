"""Centralized Streamlit UI configuration: detector registry and defaults.

Single source of truth for detector metadata (display name, description, category,
parameter ranges, and teaching-track overlay family). The sidebar sliders, the
teaching track and the results views all read from ``DETECTOR_REGISTRY``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectors.constants import DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE

DetectorCategory = Literal[
    "Density",
    "Gaussian",
    "One-Class SVM",
    "Distance",
    "Univariate",
    "Dimensionality",
    "Sequential",
]

DetectorFamily = Literal[
    "iforest",
    "covariance_empirical",
    "covariance_elliptic",
    "covariance_robust",
    "knn",
    "lof",
    "ocsvm",
    "zscore",
    "pca",
    "none",
]


@dataclass(frozen=True)
class NumParam:
    """Numeric slider parameter for a detector."""

    label: str
    min_val: float | int
    max_val: float | int
    default: float | int
    step: float | int
    arg_name: str


@dataclass(frozen=True)
class CatParam:
    """Categorical selectbox parameter for a detector."""

    label: str
    default: str
    options: tuple[str, ...]
    arg_name: str


Param = NumParam | CatParam


@dataclass(frozen=True)
class DetectorSpec:
    """UI metadata for one detector."""

    name: str
    description: str
    category: DetectorCategory
    family: DetectorFamily = "none"
    default: bool = False
    params: tuple[Param, ...] = ()


# ============================================================================
# DETECTOR REGISTRY (unified, used by sidebar + teaching track)
# ============================================================================
# DETECTOR REGISTRY (unified, used by sidebar + teaching track)
# ============================================================================

DETECTOR_REGISTRY: dict[str, DetectorSpec] = {
    "Isolation Forest": DetectorSpec(
        name="Isolation Forest",
        description=(
            "Isolates anomalies by randomly partitioning feature space, requiring fewer cuts "
            "for isolated points. Ideal for fast, scalable detection in multi-dimensional datasets."
        ),
        category="Density",
        family="iforest",
        default=True,
        params=(
            NumParam(label="Contamination", min_val=0.01, max_val=0.20, default=0.05, step=0.01, arg_name="contamination"),
            NumParam(label="Trees", min_val=20, max_val=300, default=100, step=10, arg_name="n_estimators"),
            CatParam(label="Max samples", default="auto", options=("auto", "0.2", "0.4", "0.6", "0.8", "1.0"), arg_name="max_samples"),
        ),
    ),
    "Extended IForest": DetectorSpec(
        name="Extended IForest",
        description=(
            "Cuts feature space along random oblique hyperplanes to eliminate axis-aligned "
            "slicing artifacts. Ideal for high-dimensional data with strongly correlated features."
        ),
        category="Density",
        family="iforest",
        params=(
            NumParam(label="Contamination", min_val=0.01, max_val=0.20, default=0.05, step=0.01, arg_name="contamination"),
            NumParam(label="Trees", min_val=20, max_val=300, default=100, step=10, arg_name="n_estimators"),
            CatParam(label="Max samples", default="auto", options=("auto", "0.2", "0.4", "0.6", "0.8", "1.0"), arg_name="max_samples"),
        ),
    ),
    "Mahalanobis": DetectorSpec(
        name="Mahalanobis",
        description=(
            "Measures directional distance to the mean accounting for covariance between "
            "variables. Ideal for unimodal, Gaussian-like data with correlated features."
        ),
        category="Gaussian",
        family="covariance_empirical",
        default=True,
        params=(NumParam(label="Threshold percentile", min_val=80, max_val=99, default=95, step=1, arg_name="threshold_percentile"),),
    ),
    "Elliptic Envelope": DetectorSpec(
        name="Elliptic Envelope",
        description=(
            "Fits a robust elliptical Gaussian boundary using Minimum Covariance Determinant "
            "estimation. Ideal for unimodal data when training samples contain known outliers."
        ),
        category="Gaussian",
        family="covariance_elliptic",
        default=False,
        params=(NumParam(label="Contamination", min_val=0.01, max_val=0.30, default=0.10, step=0.01, arg_name="contamination"),),
    ),
    "Robust Covariance": DetectorSpec(
        name="Robust Covariance",
        description=(
            "Computes a high-breakdown covariance estimate by evaluating only the densest "
            "subset of points. Ideal for establishing stable elliptical boundaries in noisy datasets."
        ),
        category="Gaussian",
        family="covariance_robust",
        params=(NumParam(label="Contamination", min_val=0.01, max_val=0.30, default=0.10, step=0.01, arg_name="contamination"),),
    ),
    "KNN": DetectorSpec(
        name="KNN",
        description=(
            "Scores samples based on distance to their $k$-th nearest neighbors in feature "
            "space. Ideal for non-parametric datasets without pre-assumed distribution shapes."
        ),
        category="Distance",
        family="knn",
        params=(
            NumParam(label="Neighbors", min_val=3, max_val=50, default=5, step=1, arg_name="n_neighbors"),
            NumParam(label="Contamination", min_val=0.01, max_val=0.30, default=0.10, step=0.01, arg_name="contamination"),
        ),
    ),
    "OC-SVM (RBF)": DetectorSpec(
        name="OC-SVM (RBF)",
        description=(
            "Maps data into high-dimensional space to enclose normal points within a smooth, "
            "flexible envelope. Ideal for non-linear, multi-modal distributions with complex "
            "decision boundaries."
        ),
        category="One-Class SVM",
        family="ocsvm",
        params=(
            NumParam(label="Nu", min_val=0.01, max_val=0.30, default=0.05, step=0.01, arg_name="nu"),
            CatParam(label="Gamma", default="auto", options=("auto", "scale", "0.01", "0.1", "0.5", "1.0"), arg_name="gamma"),
        ),
    ),
    "OC-SVM (Linear)": DetectorSpec(
        name="OC-SVM (Linear)",
        description=(
            "Constructs a linear decision boundary separating normal training samples from the "
            "origin. Ideal for high-dimensional, linearly separable features requiring high "
            "computational efficiency."
        ),
        category="One-Class SVM",
        family="ocsvm",
        params=(NumParam(label="Nu", min_val=0.01, max_val=0.30, default=0.05, step=0.01, arg_name="nu"),),
    ),
    "OC-SVM (Poly)": DetectorSpec(
        name="OC-SVM (Poly)",
        description=(
            "Encloses normal samples using polynomial feature interactions to construct curved "
            "decision boundaries. Ideal for structured, non-linear patterns with distinct "
            "geometric curvature."
        ),
        category="One-Class SVM",
        family="ocsvm",
        params=(
            NumParam(label="Nu", min_val=0.01, max_val=0.30, default=0.05, step=0.01, arg_name="nu"),
            CatParam(label="Gamma", default="auto", options=("auto", "scale", "0.01", "0.1", "0.5", "1.0"), arg_name="gamma"),
        ),
    ),
    "LOF": DetectorSpec(
        name="LOF",
        description=(
            "Compares local sample density against the density of its immediate "
            "$k$-nearest neighbors. Ideal for heterogeneous datasets containing clusters of "
            "varying densities."
        ),
        category="Density",
        family="lof",
        params=(
            NumParam(label="Neighbors", min_val=5, max_val=50, default=20, step=1, arg_name="n_neighbors"),
            NumParam(label="Contamination", min_val=0.01, max_val=0.30, default=0.05, step=0.01, arg_name="contamination"),
        ),
    ),
    "Z-Score": DetectorSpec(
        name="Z-Score",
        description=(
            "Measures how many standard deviations individual feature values deviate from "
            "their mean. Ideal for fast univariate screening and establishing interpretable "
            "baseline thresholds."
        ),
        category="Univariate",
        family="zscore",
        params=(NumParam(label="Threshold (std)", min_val=1.0, max_val=5.0, default=3.0, step=0.1, arg_name="threshold"),),
    ),
    "PCA Reconstruction": DetectorSpec(
        name="PCA Reconstruction",
        description=(
            "Projects data onto dominant linear subspaces and measures reconstruction error "
            "upon decoding. Ideal for continuous high-dimensional data governed by linear "
            "latent relationships."
        ),
        category="Dimensionality",
        family="pca",
        params=(
            NumParam(label="Components", min_val=1, max_val=20, default=5, step=1, arg_name="n_components"),
            NumParam(label="Threshold percentile", min_val=50, max_val=99, default=95, step=1, arg_name="threshold_percentile"),
        ),
    ),
    "HMM": DetectorSpec(
        name="HMM",
        description=(
            "Models temporal dynamics as probabilistic transitions between discrete hidden "
            "operational states. Ideal for continuous sequential data with distinct regime "
            "shifts over time."
        ),
        category="Sequential",
        params=(NumParam(label="States", min_val=2, max_val=8, default=3, step=1, arg_name="n_components"),),
    ),
    "Hawkes": DetectorSpec(
        name="Hawkes",
        description=(
            "Models self-exciting point processes where past events temporarily increase "
            "future event probability. Ideal for discrete event streams and catching "
            "localized activity bursts or cascades."
        ),
        category="Sequential",
        params=(NumParam(label="Decay", min_val=0.1, max_val=1.0, default=0.5, step=0.1, arg_name="decay"),),
    ),
    "Markov Sequence": DetectorSpec(
        name="Markov Sequence",
        description=(
            "Evaluates step-by-step state transition likelihoods against learned normal "
            "sequence patterns. Ideal for discrete event logs and identifying unexpected "
            "execution ordering or workflow shifts."
        ),
        category="Sequential",
        params=(
            NumParam(label="Threshold percentile", min_val=80, max_val=99, default=90, step=1, arg_name="threshold_percentile"),
        ),
    ),
}

# Display order = insertion order of the registry.
DETECTOR_NAMES: list[str] = list(DETECTOR_REGISTRY.keys())

DETECTOR_DEFAULTS_LIST: list[str] = [
    name for name, spec in DETECTOR_REGISTRY.items() if spec.default
]

DETECTOR_CATEGORIES: dict[str, list[str]] = {}
for spec in DETECTOR_REGISTRY.values():
    DETECTOR_CATEGORIES.setdefault(spec.category, []).append(spec.name)

# ============================================================================
# PIPELINE DEFAULTS
# ============================================================================

ENSEMBLE_DEFAULTS = {
    "mode": "soft",
    "threshold_percentile": DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
}
