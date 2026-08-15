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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Score a fitted detector over a 2-D mesh, optionally mapped to feature space.

    Mesh points start in 2-D and are passed through ``transform`` (e.g. a PCA
    ``inverse_transform``) before scoring, so the background gradient is the
    detector's actual score field on the plane, never an approximation.

    Returns ``(xx, yy, zz, threshold)``. ``threshold`` is the score value in
    the [0, 1] mesh space that separates normal from anomalous points: the
    midpoint between the highest normal score and the lowest anomaly score on
    the mesh. Reading the detector's own ``threshold`` would be wrong here —
    several detectors store it in raw score space (e.g. Z-Score stores a z
    value of 3.0), while ``zz`` is always the normalized score, so the isoline
    would fall outside the plotted range. Falls back to 0.5 if the mesh has no
    normal or no anomalous point.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)
    xx, yy = np.meshgrid(xs, ys)
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    if transform is not None:
        mesh = transform(mesh)
    grid_pred, grid_scores = detector.predict(mesh)

    normal_scores = grid_scores[grid_pred == 0]
    anomaly_scores = grid_scores[grid_pred == 1]
    if normal_scores.size and anomaly_scores.size:
        threshold = float((normal_scores.max() + anomaly_scores.min()) / 2)
    elif normal_scores.size:
        threshold = float(normal_scores.max())
    else:
        threshold = 0.5

    return xx, yy, grid_scores.reshape(xx.shape), float(threshold)


def contour_polylines(
    zz: np.ndarray,
    level: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: int = GRID_RESOLUTION,
) -> list[np.ndarray]:
    """Trace the iso-line(s) of ``zz`` at ``level`` as real-coordinate polylines.

    Marching squares on the score mesh. Drawing the boundary as plain ``go.Scatter``
    lines (instead of a single-level Plotly contour isoline, which can silently
    not render) guarantees the decision boundary is always visible.
    Returns one ``(N, 2)`` array per closed or open contour, in plot coordinates.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)

    def cross_edge(v1, p1, v2, p2):
        if (v1 >= level) == (v2 >= level):
            return None
        denom = v2 - v1
        t = 0.0 if denom == 0 else (level - v1) / denom
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    ny, nx = zz.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v00, v10 = zz[i, j], zz[i, j + 1]
            v11, v01 = zz[i + 1, j + 1], zz[i + 1, j]
            if (v00 >= level) == (v10 >= level) == (v11 >= level) == (v01 >= level):
                continue
            bl, br, tr, tl = (xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1])
            e0 = cross_edge(v00, bl, v10, br)
            e1 = cross_edge(v10, br, v11, tr)
            e2 = cross_edge(v11, tr, v01, tl)
            e3 = cross_edge(v01, tl, v00, bl)
            crossed = [e for e in (e0, e1, e2, e3) if e is not None]
            if len(crossed) == 2:
                segments.append((crossed[0], crossed[1]))
            elif len(crossed) == 4:  # saddle: connect via the cell-average decider
                if (v00 + v10 + v11 + v01) / 4.0 >= level:
                    segments.extend(((e0, e1), (e2, e3)))
                else:
                    segments.extend(((e0, e3), (e1, e2)))

    if not segments:
        return []

    def key(p: tuple[float, float]) -> tuple[float, float]:
        return (round(p[0], 6), round(p[1], 6))

    from collections import defaultdict

    adjacency: dict[tuple[float, float], list[tuple[int, int, tuple[float, float]]]] = defaultdict(list)
    for si, (a, b) in enumerate(segments):
        adjacency[key(a)].append((si, 0, a))
        adjacency[key(b)].append((si, 1, b))

    used: set[int] = set()
    polylines: list[np.ndarray] = []
    for si in range(len(segments)):
        if si in used:
            continue
        used.add(si)
        a, b = segments[si]
        chain = [a, b]
        # extend backward from a
        cur_key = key(a)
        while True:
            nxt = [e for e in adjacency[cur_key] if e[0] not in used]
            if not nxt:
                break
            nsi, nend, _ = nxt[0]
            used.add(nsi)
            oa, ob = segments[nsi]
            chain.insert(0, ob if nend == 0 else oa)
            cur_key = key(chain[0])
        # extend forward from b
        cur_key = key(b)
        while True:
            nxt = [e for e in adjacency[cur_key] if e[0] not in used]
            if not nxt:
                break
            nsi, nend, _ = nxt[0]
            used.add(nsi)
            oa, ob = segments[nsi]
            chain.append(oa if nend == 0 else ob)
            cur_key = key(chain[-1])
        if key(chain[-1]) == key(chain[0]):  # closed loop: close the ring
            chain.append(chain[0])
        polylines.append(np.asarray(chain))

    return polylines
