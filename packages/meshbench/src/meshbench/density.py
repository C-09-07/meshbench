"""Polycount efficiency: vertices per area, face density variance."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.types import DensityReport


def vertices_per_area(mesh: trimesh.Trimesh) -> float:
    """Compute vertex count / surface area."""
    area = mesh.area
    if area < 1e-12:
        return 0.0
    return mesh.vertices.shape[0] / area


def face_density_variance(mesh: trimesh.Trimesh) -> float:
    """Normalized variance of per-face area.

    Returns 0.0 for perfectly uniform face sizes. Higher values
    indicate more variation in triangle sizes across the mesh.
    Normalized by mean^2 to be scale-independent (coefficient of variation squared).
    """
    if mesh.faces.shape[0] < 2:
        return 0.0

    areas = mesh.area_faces
    mean = float(np.mean(areas))
    if mean < 1e-12:
        return 0.0

    variance = float(np.var(areas))
    return variance / (mean * mean)


def compute_density_report(mesh: trimesh.Trimesh) -> DensityReport:
    """Compute a full DensityReport for a mesh."""
    return DensityReport(
        vertices_per_area=round(vertices_per_area(mesh), 4),
        face_density_variance=round(face_density_variance(mesh), 6),
    )
