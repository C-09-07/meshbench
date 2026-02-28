"""Adaptive remeshing: curvature-weighted decimation.

Both decimate() and adaptive_remesh() use per-vertex quality weighting via
pymeshlab when available. adaptive_remesh() adds the curvature_weight parameter
to control how strongly curvature biases the importance map (higher = more
aggressive preservation of curved regions).

Without pymeshlab, falls back to uniform QEM via fast-simplification.
"""

from __future__ import annotations

import time

import numpy as np
import trimesh

from meshbench.features import compute_feature_report
from meshfix._backends import _HAS_PYMESHLAB
from meshfix._decimate import (
    _decimate_pymeshlab,
    _decimate_trimesh,
    _resolve_target,
    _vertex_importance,
)
from meshfix.types import Backend, DecimateConfig, DecimateResult


def adaptive_remesh(
    mesh: trimesh.Trimesh,
    target_faces: int | None = None,
    target_ratio: float | None = None,
    curvature_weight: float = 1.0,
    feature_angle: float = 30.0,
) -> tuple[trimesh.Trimesh, DecimateResult]:
    """Curvature-weighted decimation: fewer faces on flat, more on curved.

    With pymeshlab: per-vertex quality weighting biases QEM to preserve
    high-curvature vertices and feature edges. curvature_weight scales the
    curvature term (higher = stronger preservation of curved regions).

    Without pymeshlab: falls back to uniform QEM via fast-simplification.

    Does not mutate the input mesh.
    """
    target = _resolve_target(mesh, target_faces, target_ratio)
    n_faces = mesh.faces.shape[0]

    if target >= n_faces:
        feat = compute_feature_report(mesh, feature_angle=feature_angle)
        out = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )
        return out, DecimateResult(
            backend=Backend.TRIMESH,
            faces_before=n_faces,
            faces_after=n_faces,
            vertices_before=mesh.vertices.shape[0],
            vertices_after=mesh.vertices.shape[0],
            feature_edges_before=feat.feature_edge_count,
            feature_edges_after=feat.feature_edge_count,
            duration_ms=0.0,
        )

    feat_before = compute_feature_report(mesh, feature_angle=feature_angle)
    t0 = time.perf_counter()

    if _HAS_PYMESHLAB:
        backend = Backend.PYMESHLAB
        importance = _vertex_importance(mesh, feature_angle, curvature_weight)
        config = DecimateConfig(
            feature_angle=feature_angle, preserve_boundary=True,
        )
        result = _decimate_pymeshlab(mesh, target, config, importance=importance)
    else:
        backend = Backend.TRIMESH
        result = _decimate_trimesh(mesh, target)

    duration = (time.perf_counter() - t0) * 1000.0
    feat_after = compute_feature_report(result, feature_angle=feature_angle)

    return result, DecimateResult(
        backend=backend,
        faces_before=n_faces,
        faces_after=result.faces.shape[0],
        vertices_before=mesh.vertices.shape[0],
        vertices_after=result.vertices.shape[0],
        feature_edges_before=feat_before.feature_edge_count,
        feature_edges_after=feat_after.feature_edge_count,
        duration_ms=duration,
    )
