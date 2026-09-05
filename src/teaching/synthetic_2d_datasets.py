"""Didactic synthetic dataset generator for anomaly teaching.

Includes:
- 2-D scikit-learn datasets (blobs, moons, circles, swiss_roll) for the teaching track
- Synthetic CASAS-style event stream generator (asymmetric Markov chain) for the CASAS track
"""

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons, make_swiss_roll

from detectors.constants import DEFAULT_RANDOM_STATE

# Small uniform mix so the graph's structure dominates (else a reversal has nothing rare).
NOISE = 0.10


def _make_blobs_inliers(n: int, rng: np.random.RandomState) -> np.ndarray:
    return np.asarray(
        make_blobs(
            n, centers=[[-5, -5], [0, 0], [5, 5]], cluster_std=1.0, random_state=rng
        )[0]
    )


def _make_moons_inliers(n: int, rng: np.random.RandomState) -> np.ndarray:
    return np.asarray(make_moons(n, noise=0.08, random_state=rng)[0])


def _make_circles_inliers(n: int, rng: np.random.RandomState) -> np.ndarray:
    return np.asarray(make_circles(n, noise=0.05, factor=0.5, random_state=rng)[0])


def _make_swiss_roll_inliers(n: int, rng: np.random.RandomState) -> np.ndarray:
    X_3d = np.asarray(make_swiss_roll(n, noise=0.5, random_state=rng)[0])
    return X_3d[:, [0, 2]]


class SyntheticDatasetGenerator:
    """Generate synthetic datasets with known characteristics for visualization."""

    DATASETS: ClassVar[dict[str, str]] = {
        "Blobs (Gaussian)": "blobs",
        "Moons (Non-linear)": "moons",
        "Circles (Concentric)": "circles",
        "Swiss Roll (Manifold)": "swiss_roll",
    }

    _REGISTRY: ClassVar[
        dict[str, tuple[Callable[[int, np.random.RandomState], np.ndarray], tuple[float, float]]]
    ] = {
        "blobs": (_make_blobs_inliers, (-8, 8)),
        "moons": (_make_moons_inliers, (-2, 3)),
        "circles": (_make_circles_inliers, (-2, 2)),
        "swiss_roll": (_make_swiss_roll_inliers, (-15, 15)),
    }

    @classmethod
    def generate(
        cls,
        dataset_name: str,
        n_samples: int = 300,
        contamination: float = 0.1,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic dataset with injected anomalies.

        Returns:
            X: (n_samples, 2) - 2D data
            y_true: (n_samples,) - true labels (0=normal, 1=anomaly)
        """
        if dataset_name not in cls._REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        rng = np.random.RandomState(random_state)
        gen_fn, (lo, hi) = cls._REGISTRY[dataset_name]
        n_inliers = int(n_samples * (1 - contamination))
        n_outliers = n_samples - n_inliers

        X_inliers = gen_fn(n_inliers, rng)
        X_outliers = rng.uniform(lo, hi, size=(n_outliers, X_inliers.shape[1]))
        X = np.vstack([X_inliers, X_outliers])
        y = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])
        perm = rng.permutation(len(X))
        return X[perm], y[perm]


def sensor_chain_probabilities(n_sensors: int) -> np.ndarray:
    """First-order transition matrix: asymmetric directed cycle (``i -> i+1`` dominant).

    Asymmetry makes a reversed day (collective injector) produce rare transitions;
    a symmetric stream would be indistinguishable after reversal. Falls back to
    uniform for ``n_sensors < 3`` (a cycle needs three states).
    """
    if n_sensors < 3:
        return np.full((n_sensors, n_sensors), 1.0 / n_sensors)
    weights = np.ones((n_sensors, n_sensors))
    for i in range(n_sensors):
        weights[i, i] = 5.0
        weights[i, (i + 1) % n_sensors] = 25.0
        weights[i, (i - 1) % n_sensors] = 2.0
    probs = weights / weights.sum(axis=1, keepdims=True)
    return (1.0 - NOISE) * probs + NOISE / n_sensors


def generate_synthetic_events(
    n_days: int = 5,
    pattern: str = "regular",
    n_sensors: int = 3,
    events_per_day: int = 80,
    seed: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Synthetic event stream in CASAS-Aruba schema (no DB needed).

    Sensors are drawn from an asymmetric first-order Markov chain (see
    ``sensor_chain_probabilities``) so the sequence extractors have structure to
    learn and a reversal produces rare transitions.

    pattern: "regular" (evenly spaced), "bursty" (temporal clusters), or
    "day_night" (mostly daytime).
    """
    rng = np.random.default_rng(seed)
    sensors = [f"Sensor_{i + 1}" for i in range(n_sensors)]
    transition_probs = sensor_chain_probabilities(n_sensors)
    base_day = pd.Timestamp("2024-01-01")
    records = []

    for day in range(n_days):
        day_start = base_day + pd.Timedelta(days=day)

        if pattern == "regular":
            offsets = np.linspace(0, 24 * 60, events_per_day, endpoint=False)
            offsets = offsets + rng.normal(0, 2, size=events_per_day)
        elif pattern == "bursty":
            n_clusters = max(3, events_per_day // 15)
            centers = rng.uniform(0, 24 * 60, size=n_clusters)
            per_cluster = max(events_per_day // n_clusters, 1)
            offsets = np.concatenate(
                [rng.normal(center, 5, size=per_cluster) for center in centers]
            )
        elif pattern == "day_night":
            day_events = int(events_per_day * 0.85)
            night_events = events_per_day - day_events
            offsets = np.concatenate(
                [
                    rng.uniform(8 * 60, 22 * 60, size=day_events),
                    rng.uniform(0, 8 * 60, size=night_events // 2),
                    rng.uniform(
                        22 * 60, 24 * 60, size=night_events - night_events // 2
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        offsets = np.clip(offsets, 0, 24 * 60 - 0.01)
        timestamps = [day_start + pd.Timedelta(minutes=float(minutes)) for minutes in offsets]
        seq = np.empty(len(timestamps), dtype=int)
        seq[0] = int(rng.integers(0, n_sensors))
        for i in range(1, len(timestamps)):
            seq[i] = int(rng.choice(n_sensors, p=transition_probs[seq[i - 1]]))
        chosen_sensors = [sensors[int(i)] for i in seq]
        event_types = rng.choice(["ON", "OFF"], size=len(timestamps))

        for timestamp, sensor, event_type in zip(
            timestamps, chosen_sensors, event_types, strict=True
        ):
            records.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor,
                    "event_type": event_type,
                    "value": 1.0 if event_type == "ON" else 0.0,
                }
            )

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
