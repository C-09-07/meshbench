"""Feature edge detection via dihedral angle classification."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._edges import EdgeFaceMap, build_edge_face_map, edge_data_from_map
from meshbench.types import FeatureReport


def dihedral_angles(
    mesh: trimesh.Trimesh,
    _edge_face_map: EdgeFaceMap | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dihedral angle at each interior edge.

    Returns:
        (edges, angles) where edges is (N, 2) vertex pairs and
        angles is (N,) in degrees [0, 180]. Only interior edges
        (shared by exactly 2 faces) are included.
    """
    if mesh.faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    if _edge_face_map is None:
        _edge_face_map = build_edge_face_map(mesh.faces)

    sorted_edges, sorted_face_ids, edge_starts, edge_ends = _edge_face_map
    counts = edge_ends - edge_starts

    # Interior edges: shared by exactly 2 faces
    interior_mask = counts == 2
    interior_starts = edge_starts[interior_mask]

    if len(interior_starts) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    # Get the two face indices per interior edge
    f0 = sorted_face_ids[interior_starts]
    f1 = sorted_face_ids[interior_starts + 1]

    # Face normals
    normals = mesh.face_normals
    n0 = normals[f0]
    n1 = normals[f1]

    # Dihedral angle: arccos(dot(n0, n1))
    # Coplanar → dot≈1 → angle≈0°. Perpendicular → angle≈90°.
    dots = np.sum(n0 * n1, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    edges = sorted_edges[edge_starts[interior_mask]]
    return edges, angles


def feature_edges(
    mesh: trimesh.Trimesh,
    feature_angle: float = 30.0,
    _edge_face_map: EdgeFaceMap | None = None,
) -> np.ndarray:
    """Return (N, 2) vertex pairs where dihedral angle > feature_angle."""
    edges, angles = dihedral_angles(mesh, _edge_face_map=_edge_face_map)
    if len(angles) == 0:
        return np.empty((0, 2), dtype=np.int64)
    mask = angles > feature_angle
    return edges[mask]


def compute_feature_report(
    mesh: trimesh.Trimesh,
    feature_angle: float = 30.0,
    include_angles: bool = False,
    _edge_face_map: EdgeFaceMap | None = None,
) -> FeatureReport:
    """Compute feature edge report with dihedral angle classification."""
    if mesh.faces.shape[0] == 0:
        return FeatureReport(
            feature_edge_count=0,
            smooth_edge_count=0,
            boundary_edge_count=0,
            feature_angle_deg=feature_angle,
            feature_edge_ratio=0.0,
            dihedral_angles=None,
        )

    if _edge_face_map is None:
        _edge_face_map = build_edge_face_map(mesh.faces)

    _, _, edge_starts, edge_ends = _edge_face_map
    counts = edge_ends - edge_starts

    boundary_count = int(np.sum(counts == 1))
    interior_count = int(np.sum(counts == 2))
    # Non-manifold edges (>2) are counted separately
    nm_count = int(np.sum(counts > 2))

    edges, angles = dihedral_angles(mesh, _edge_face_map=_edge_face_map)
    if len(angles) == 0:
        return FeatureReport(
            feature_edge_count=0,
            smooth_edge_count=0,
            boundary_edge_count=boundary_count,
            feature_angle_deg=feature_angle,
            feature_edge_ratio=0.0,
            dihedral_angles=None,
        )

    feature_mask = angles > feature_angle
    feature_count = int(np.sum(feature_mask))
    smooth_count = int(np.sum(~feature_mask))

    total_interior = interior_count + nm_count
    ratio = feature_count / total_interior if total_interior > 0 else 0.0

    return FeatureReport(
        feature_edge_count=feature_count,
        smooth_edge_count=smooth_count,
        boundary_edge_count=boundary_count,
        feature_angle_deg=feature_angle,
        feature_edge_ratio=ratio,
        dihedral_angles=angles if include_angles else None,
    )
