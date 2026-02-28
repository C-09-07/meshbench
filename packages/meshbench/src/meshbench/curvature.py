"""Discrete curvature estimation: mean (cotangent Laplacian) and Gaussian (angle deficit)."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.types import CurvatureReport


def _cotangent_weights_and_areas(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cotangent Laplacian and mixed Voronoi vertex areas.

    Returns:
        (laplacian_norm, vertex_areas) where laplacian_norm is (V,) magnitude
        of the Laplacian vector and vertex_areas is (V,) mixed Voronoi area.
    """
    n_verts = mesh.vertices.shape[0]
    faces = mesh.faces
    verts = mesh.vertices

    v = verts[faces]  # (F, 3, 3)
    # Edge vectors per face
    e0 = v[:, 1] - v[:, 0]  # opposite vertex 2
    e1 = v[:, 2] - v[:, 1]  # opposite vertex 0
    e2 = v[:, 0] - v[:, 2]  # opposite vertex 1

    # Cotangent at each vertex: cot(θ) = dot(a,b) / |cross(a,b)|
    # At v0: angle between e0 and -e2
    cross0 = np.linalg.norm(np.cross(e0, -e2), axis=1)
    dot0 = np.sum(e0 * (-e2), axis=1)
    # At v1: angle between -e0 and e1
    cross1 = np.linalg.norm(np.cross(-e0, e1), axis=1)
    dot1 = np.sum((-e0) * e1, axis=1)
    # At v2: angle between -e1 and e2
    cross2 = np.linalg.norm(np.cross(-e1, e2), axis=1)
    dot2 = np.sum((-e1) * e2, axis=1)

    # Clamp cross to avoid division by zero
    eps = 1e-12
    cross0 = np.maximum(cross0, eps)
    cross1 = np.maximum(cross1, eps)
    cross2 = np.maximum(cross2, eps)

    cot0 = dot0 / cross0
    cot1 = dot1 / cross1
    cot2 = dot2 / cross2

    # Laplacian: L(vi) = sum over edges (vi,vj) of cot(α_ij) + cot(β_ij) * (vj - vi)
    # For edge (v0, v1) in a face, the opposite angle is at v2 → weight is cot2
    # For edge (v1, v2), opposite is v0 → weight is cot0
    # For edge (v2, v0), opposite is v1 → weight is cot1
    laplacian = np.zeros((n_verts, 3), dtype=np.float64)
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Edge (v0→v1): weight cot2, contributes (v1-v0) to v0 and (v0-v1) to v1
    w01 = cot2[:, np.newaxis]
    diff01 = v[:, 1] - v[:, 0]
    np.add.at(laplacian, f0, w01 * diff01)
    np.add.at(laplacian, f1, -w01 * diff01)

    # Edge (v1→v2): weight cot0
    w12 = cot0[:, np.newaxis]
    diff12 = v[:, 2] - v[:, 1]
    np.add.at(laplacian, f1, w12 * diff12)
    np.add.at(laplacian, f2, -w12 * diff12)

    # Edge (v2→v0): weight cot1
    w20 = cot1[:, np.newaxis]
    diff20 = v[:, 0] - v[:, 2]
    np.add.at(laplacian, f2, w20 * diff20)
    np.add.at(laplacian, f0, -w20 * diff20)

    # Vertex areas: approximate as face_area / 3 per vertex
    face_areas = mesh.area_faces
    vertex_areas = np.zeros(n_verts, dtype=np.float64)
    area_third = face_areas / 3.0
    np.add.at(vertex_areas, f0, area_third)
    np.add.at(vertex_areas, f1, area_third)
    np.add.at(vertex_areas, f2, area_third)

    # |H| = |L| / (2 * A)
    vertex_areas = np.maximum(vertex_areas, eps)
    laplacian_norm = np.linalg.norm(laplacian, axis=1) / (2.0 * vertex_areas)

    return laplacian_norm, vertex_areas


def discrete_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex |H| via cotangent Laplacian. Returns (V,)."""
    if mesh.faces.shape[0] == 0:
        return np.zeros(mesh.vertices.shape[0], dtype=np.float64)
    h, _ = _cotangent_weights_and_areas(mesh)
    return h


def discrete_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex K via angle deficit. Returns (V,).

    K_i = (2π - sum of angles at vertex i) / A_i
    """
    n_verts = mesh.vertices.shape[0]
    if mesh.faces.shape[0] == 0:
        return np.zeros(n_verts, dtype=np.float64)

    faces = mesh.faces
    v = mesh.vertices[faces]  # (F, 3, 3)

    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 1]
    e2 = v[:, 0] - v[:, 2]

    def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = na * nb
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.sum(a * b, axis=1) / denom
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.arccos(cos)
        ang[denom == 0] = 0.0
        return ang

    # Angles at each vertex
    a0 = _angles(e0, -e2)  # at v0
    a1 = _angles(-e0, e1)  # at v1
    a2 = _angles(-e1, e2)  # at v2

    # Sum angles per vertex
    angle_sum = np.zeros(n_verts, dtype=np.float64)
    np.add.at(angle_sum, faces[:, 0], a0)
    np.add.at(angle_sum, faces[:, 1], a1)
    np.add.at(angle_sum, faces[:, 2], a2)

    # Vertex areas
    eps = 1e-12
    face_areas = mesh.area_faces
    vertex_areas = np.zeros(n_verts, dtype=np.float64)
    area_third = face_areas / 3.0
    np.add.at(vertex_areas, faces[:, 0], area_third)
    np.add.at(vertex_areas, faces[:, 1], area_third)
    np.add.at(vertex_areas, faces[:, 2], area_third)

    vertex_areas = np.maximum(vertex_areas, eps)
    K = (2.0 * np.pi - angle_sum) / vertex_areas

    # Zero out isolated vertices (not in any face)
    in_face = np.zeros(n_verts, dtype=bool)
    in_face[faces.ravel()] = True
    K[~in_face] = 0.0

    return K


def per_face_curvature(
    mesh: trimesh.Trimesh,
    _vertex_curvature: np.ndarray | None = None,
) -> np.ndarray:
    """Average vertex curvature to faces. Returns (F,).

    Uses mean curvature |H| by default, or accepts pre-computed per-vertex values.
    """
    if mesh.faces.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if _vertex_curvature is None:
        _vertex_curvature = discrete_mean_curvature(mesh)
    face_curv = _vertex_curvature[mesh.faces]  # (F, 3)
    return np.mean(face_curv, axis=1)


def compute_curvature_report(mesh: trimesh.Trimesh) -> CurvatureReport:
    """Compute discrete curvature statistics."""
    if mesh.faces.shape[0] == 0:
        return CurvatureReport(
            mean_curvature_abs_avg=0.0,
            mean_curvature_abs_max=0.0,
            gaussian_curvature_avg=0.0,
            gaussian_curvature_std=0.0,
            flat_vertex_ratio=1.0,
        )

    H = discrete_mean_curvature(mesh)
    K = discrete_gaussian_curvature(mesh)

    # Only consider vertices actually in faces
    in_face = np.zeros(mesh.vertices.shape[0], dtype=bool)
    in_face[mesh.faces.ravel()] = True
    H_active = H[in_face]
    K_active = K[in_face]

    eps = 1e-6
    flat_count = int(np.sum(H_active < eps))
    flat_ratio = flat_count / len(H_active) if len(H_active) > 0 else 1.0

    return CurvatureReport(
        mean_curvature_abs_avg=float(np.mean(H_active)) if len(H_active) > 0 else 0.0,
        mean_curvature_abs_max=float(np.max(H_active)) if len(H_active) > 0 else 0.0,
        gaussian_curvature_avg=float(np.mean(K_active)) if len(K_active) > 0 else 0.0,
        gaussian_curvature_std=float(np.std(K_active)) if len(K_active) > 0 else 0.0,
        flat_vertex_ratio=flat_ratio,
    )
