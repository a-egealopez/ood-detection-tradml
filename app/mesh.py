"""Shared mesh-scoring helper for decision-boundary charts.

Both teaching charts (2D Playground, CASAS score clouds) draw the detector's
real anomaly-score field over a grid. The only difference is whether the mesh
must be mapped back to feature space before scoring (CASAS: PCA plane -> 9-D
features). This helper centralizes that so the two views cannot drift apart.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

GRID_RESOLUTION = 60


def score_mesh(
    detector: Any,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: int = GRID_RESOLUTION,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a fitted detector over a 2-D mesh, optionally mapped to feature space.

    Mesh points start in 2-D and are passed through ``transform`` (e.g. a PCA
    ``inverse_transform``) before scoring, so the background gradient is the
    detector's actual score field on the plane, never an approximation.

    Returns ``(xx, yy, zz)`` with ``zz`` the anomaly scores reshaped to the grid.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)
    xx, yy = np.meshgrid(xs, ys)
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    if transform is not None:
        mesh = transform(mesh)
    _, grid_scores = detector.predict(mesh)
    return xx, yy, grid_scores.reshape(xx.shape)
