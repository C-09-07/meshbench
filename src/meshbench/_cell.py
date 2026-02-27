"""Per-face (cell) quality metrics: aspect ratio and minimum angle."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.types import CellReport


def _edge_vectors(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three edge vectors per face.

    Returns (e0, e1, e2) each of shape (F, 3) where:
        e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2
    """
    v = mesh.vertices[mesh.faces]  # (F, 3, 3)
    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 1]
    e2 = v[:, 0] - v[:, 2]
    return e0, e1, e2


def _edge_lengths_squared(
    e0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return squared edge lengths per face."""
    return (
        np.sum(e0 * e0, axis=1),
        np.sum(e1 * e1, axis=1),
        np.sum(e2 * e2, axis=1),
    )


def per_face_aspect_ratio(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute aspect ratio per face: longest_edge^2 / (2 * area).

    Equilateral triangle: ~1.15. Degenerate (zero area): inf.
    """
    e0, e1, e2 = _edge_vectors(mesh)
    l0sq, l1sq, l2sq = _edge_lengths_squared(e0, e1, e2)
    longest_sq = np.maximum(np.maximum(l0sq, l1sq), l2sq)
    areas = mesh.area_faces
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = longest_sq / (2.0 * areas)
    ar[areas == 0] = np.inf
    return ar


def per_face_min_angle(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute minimum interior angle (degrees) per face.

    Uses arccos of dot products between adjacent edges.
    """
    e0, e1, e2 = _edge_vectors(mesh)
    # Interior angles at each vertex:
    # angle at v0: between edges (v1-v0) and (v2-v0) = e0 and -e2
    # angle at v1: between edges (v0-v1) and (v2-v1) = -e0 and e1
    # angle at v2: between edges (v0-v2) and (v1-v2) = e2 and -e1

    def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        na = np.sqrt(np.sum(a * a, axis=1))
        nb = np.sqrt(np.sum(b * b, axis=1))
        denom = na * nb
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.sum(a * b, axis=1) / denom
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        ang[denom == 0] = 0.0
        return ang

    a0 = _angle(e0, -e2)
    a1 = _angle(-e0, e1)
    a2 = _angle(e2, -e1)

    return np.minimum(np.minimum(a0, a1), a2)


def compute_cell_report(mesh: trimesh.Trimesh) -> CellReport:
    """Compute per-face quality metrics."""
    ar = per_face_aspect_ratio(mesh)
    angles = per_face_min_angle(mesh)

    # For aspect ratio stats, treat inf as finite max for percentile
    ar_finite = ar[np.isfinite(ar)]
    if len(ar_finite) == 0:
        ar_finite = np.array([np.inf])

    return CellReport(
        aspect_ratio_mean=float(np.mean(ar_finite)),
        aspect_ratio_std=float(np.std(ar_finite)),
        aspect_ratio_max=float(np.max(ar)),
        aspect_ratio_p95=float(np.percentile(ar_finite, 95)),
        min_angle_mean=float(np.mean(angles)),
        min_angle_std=float(np.std(angles)),
        min_angle_min=float(np.min(angles)),
        min_angle_p05=float(np.percentile(angles, 5)),
    )
