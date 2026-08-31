"""
Didactic synthetic dataset generator for anomaly teaching.
Uses scikit-learn to generate data with known geometries.
"""

from typing import ClassVar

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    make_swiss_roll,
)

from detectors.constants import DEFAULT_RANDOM_STATE


class SyntheticDatasetGenerator:
    """Generate synthetic datasets with known characteristics for visualization."""

    DATASETS: ClassVar[dict[str, str]] = {
        "Blobs (Gaussianas)": "blobs",
        "Moons (No-lineal)": "moons",
        "Circles (Anidados)": "circles",
        "Swiss Roll (Variedad)": "swiss_roll",
    }

    @staticmethod
    def generate(
        dataset_name: str,
        n_samples: int = 300,
        contamination: float = 0.1,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic dataset with injected anomalies.

        Returns:
            X: (n_samples, 2) - 2D data
            y_true: (n_samples,) - true labels (0=normal, 1=anomaly)
        """

        rng = np.random.RandomState(random_state)

        if dataset_name == "blobs":
            return SyntheticDatasetGenerator._blobs(n_samples, contamination, rng)
        elif dataset_name == "moons":
            return SyntheticDatasetGenerator._moons(n_samples, contamination, rng)
        elif dataset_name == "circles":
            return SyntheticDatasetGenerator._circles(n_samples, contamination, rng)
        elif dataset_name == "swiss_roll":
            return SyntheticDatasetGenerator._swiss_roll(n_samples, contamination, rng)
        else:
            raise ValueError(f"Dataset desconocido: {dataset_name}")

    @staticmethod
    def _blobs(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Three Gaussian clusters + random anomalies."""
        centers = np.array([[-5, -5], [0, 0], [5, 5]])
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_inliers, _ = make_blobs(
            n_samples=n_inliers,
            centers=centers,
            cluster_std=1.0,
            random_state=rng,
        )

        X_outliers = rng.uniform(-8, 8, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        return X, y_true

    @staticmethod
    def _moons(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Two interleaved half-moons + anomalies."""
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_inliers, _ = make_moons(n_samples=n_inliers, noise=0.08, random_state=rng)

        X_outliers = rng.uniform(-2, 3, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        return X, y_true

    @staticmethod
    def _circles(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Concentric circles + anomalies."""
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_inliers, _ = make_circles(
            n_samples=n_inliers,
            noise=0.05,
            factor=0.5,
            random_state=rng,
        )

        X_outliers = rng.uniform(-2, 2, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        return X, y_true

    @staticmethod
    def _swiss_roll(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Swiss roll (3D manifold projected to 2D) + anomalies."""
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_3d, _ = make_swiss_roll(n_samples=n_inliers, noise=0.5, random_state=rng)
        # Project the 3D manifold to 2D using dimensions 0 and 2.
        X_inliers = X_3d[:, [0, 2]]

        X_outliers = rng.uniform(-15, 15, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        return X, y_true
