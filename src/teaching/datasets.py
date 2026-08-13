"""
Generador de datasets sintéticos didácticos para enseñanza de anomalías.
Utiliza scikit-learn para generar datos con geometrías conocidas.
"""

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    make_swiss_roll,
)
from typing import ClassVar


class SyntheticDatasetGenerator:
    """Genera datasets sintéticos con características conocidas para visualización."""

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
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Genera dataset sintético con anomalías inyectadas.

        Returns:
            X: (n_samples, 2) - datos 2D
            y_true: (n_samples,) - etiquetas reales (0=normal, 1=anomalía)
            description: str - descripción del dataset
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
        """3 clusters gaussianos + anomalías aleatorias."""
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

        desc = f"3 clusters gaussianos con {contamination * 100:.0f}% ruido aleatorio"
        return X, y_true, desc

    @staticmethod
    def _moons(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Dos semi-lunas entrecruzadas + anomalías."""
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_inliers, _ = make_moons(n_samples=n_inliers, noise=0.08, random_state=rng)

        X_outliers = rng.uniform(-2, 3, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        desc = f"Dos semi-lunas con {contamination * 100:.0f}% puntos aislados"
        return X, y_true, desc

    @staticmethod
    def _circles(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Círculos concéntricos + anomalías."""
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

        desc = f"Círculos anidados con {contamination * 100:.0f}% puntos interiores"
        return X, y_true, desc

    @staticmethod
    def _swiss_roll(
        n_samples: int, contamination: float, rng: np.random.RandomState
    ) -> tuple:
        """Variedad suiza (manifold 3D proyectada a 2D) + anomalías."""
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_3d, _ = make_swiss_roll(n_samples=n_inliers, noise=0.5, random_state=rng)
        # Proyectar a 2D usando las primeras 2 dims
        X_inliers = X_3d[:, [0, 2]]

        X_outliers = rng.uniform(-15, 15, size=(n_outliers, 2))

        X = np.vstack([X_inliers, X_outliers])
        y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

        perm = rng.permutation(len(X))
        X, y_true = X[perm], y_true[perm]

        desc = f"Variedad suiza con {contamination * 100:.0f}% ruido uniforme"
        return X, y_true, desc


if __name__ == "__main__":
    gen = SyntheticDatasetGenerator()
    for name, key in gen.DATASETS.items():
        X, y, desc = gen.generate(key, n_samples=300, contamination=0.1)
        print(f"{name}: {desc}")
        print(f"  Shape: {X.shape}, Anomalías: {y.sum()} / {len(y)}\n")
