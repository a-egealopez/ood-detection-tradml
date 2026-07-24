"""Configuración centralizada para Streamlit app - Detectores & Ensemble."""

import sys
from pathlib import Path

# Asegurar que src está en el path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# DETECTOR DEFAULTS & RANGES
# ============================================================================

DETECTOR_PARAMS = {
    "Isolation Forest": {
        "contamination": {"min": 0.01, "max": 0.20, "default": 0.05, "step": 0.01},
    },
    "Extended IForest": {
        "contamination": {"min": 0.01, "max": 0.20, "default": 0.05, "step": 0.01},
    },
    "Mahalanobis": {
        "threshold_percentile": {"min": 80, "max": 99, "default": 95, "step": 1},
    },
    "Elliptic Envelope": {
        "contamination": {"min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
    },
    "Robust Covariance": {
        "contamination": {"min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
    },
    "KNN": {
        "n_neighbors": {"min": 3, "max": 50, "default": 5, "step": 1},
    },
    "OC-SVM": {
        "nu": {"min": 0.01, "max": 0.3, "default": 0.05, "step": 0.01},
    },
    "LOF": {
        "n_neighbors": {"min": 5, "max": 50, "default": 20, "step": 1},
    },
    "HMM": {
        "n_components": {"min": 2, "max": 8, "default": 3, "step": 1},
    },
    "Hawkes": {
        "decay": {"min": 0.1, "max": 1.0, "default": 0.5, "step": 0.1},
    },
}

ENSEMBLE_DEFAULTS = {
    "mode": "soft",
    "threshold_percentile": 90,
    "weighting_scheme": "Uniform",
}

DETECTOR_DEFAULTS_LIST = ["Isolation Forest", "Mahalanobis", "Elliptic Envelope"]

TRAINING_DEFAULTS = {
    "train_split": 0.7,
}

SYNTHETIC_EVALUATION_DEFAULTS = {
    "contamination": 0.15,
    "magnitude": 6.0,
}

# ============================================================================
# UI PRESENTATION
# ============================================================================

DETECTOR_DISPLAY_ORDER = [
    "Isolation Forest",
    "Extended IForest",
    "Mahalanobis",
    "Elliptic Envelope",
    "Robust Covariance",
    "KNN",
    "OC-SVM",
    "LOF",
    "HMM",
    "Hawkes",
]

DETECTOR_CATEGORIES = {
    "Vectorial - Densidad": ["Isolation Forest", "Extended IForest"],
    "Vectorial - Distancia": ["Mahalanobis", "Elliptic Envelope", "Robust Covariance", "KNN", "LOF"],
    "Vectorial - Frontera": ["OC-SVM"],
    "Secuencial": ["HMM", "Hawkes"],
}

DETECTOR_DESCRIPTIONS = {
    "Isolation Forest": "Aislamiento por splits aleatorios. Rápido, robusto altas dimensiones.",
    "Extended IForest": "IForest con sliced paths. Mejor en datos ultra-dimensionales.",
    "Mahalanobis": "Distancia multivariante correlada. Asume gaussiana.",
    "Elliptic Envelope": "Gaussiana robusta (MCD). Resiste outliers en fit.",
    "Robust Covariance": "Mínimo Determinante Covarianza explícito. Covarianza muy robusta.",
    "KNN": "k-vecinos más lejanos. Sin asumir distribución, agnóstico.",
    "OC-SVM": "Separador hiperplano. No-convexo, kernels soportados.",
    "LOF": "Factor de anomalía local. Densidad local vs global.",
    "HMM": "Cadena de Markov oculta. Transiciones de régimen temporal.",
    "Hawkes": "Procesos de Hawkes. Eventos con auto-excitación.",
}

# ============================================================================
# DETECTOR FACTORY MAPPING
# ============================================================================

DETECTOR_PARAM_MAP = {
    "Isolation Forest": {"contamination": "contamination"},
    "Extended IForest": {"contamination": "contamination"},
    "Mahalanobis": {"threshold_percentile": "threshold_percentile"},
    "Elliptic Envelope": {"contamination": "contamination"},
    "Robust Covariance": {"contamination": "contamination"},
    "KNN": {"n_neighbors": "n_neighbors"},
    "OC-SVM": {"nu": "nu"},
    "LOF": {"n_neighbors": "n_neighbors"},
    "HMM": {"n_components": "n_components"},
    "Hawkes": {"decay": "decay"},
}