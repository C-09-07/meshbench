"""QEM decimation with feature edge preservation."""

from __future__ import annotations

import time

import numpy as np
import trimesh

from meshbench.curvature import discrete_mean_curvature
from meshbench.features import compute_feature_report, feature_edges
from meshfix._backends import _HAS_PYMESHLAB
from meshfix.types import Backend, DecimateConfig, DecimateResult


def _resolve_target(
    mesh: trimesh.Trimesh,
    target_faces: int | None,
    target_ratio: float | None,
) -> int:
    """Resolve target face count from either absolute or ratio."""
    n = mesh.faces.shape[0]
    if target_faces is not None and target_ratio is not None:
        raise ValueError("Specify target_faces or target_ratio, not both")
    if target_faces is not None:
        return max(1, target_faces)
    if target_ratio is not None:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
        return max(1, int(n * target_ratio))
    raise ValueError("Must specify target_faces or target_ratio")


def _vertex_importance(
    mesh: trimesh.Trimesh,
    feature_angle: float,
    curvature_weight: float = 1.0,
) -> np.ndarray:
    """Per-vertex importance [0, 1] from curvature + feature edge proximity."""
    n_verts = mesh.vertices.shape[0]
    if mesh.faces.shape[0] == 0:
        return np.zeros(n_verts, dtype=np.float64)

    H = discrete_mean_curvature(mesh)
    H_max = np.max(H)
    H_norm = H / H_max if H_max > 1e-12 else np.zeros_like(H)

    feat_flag = np.zeros(n_verts, dtype=np.float64)
    feat = feature_edges(mesh, feature_angle=feature_angle)
    if len(feat) > 0:
        feat_flag[np.unique(feat.ravel())] = 1.0

    return np.clip(curvature_weight * H_norm + feat_flag, 0.0, 1.0)


def _decimate_pymeshlab(
    mesh: trimesh.Trimesh,
    target: int,
    config: DecimateConfig,
    importance: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Decimate using pymeshlab QEM with optional quality weighting."""
    import pymeshlab

    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)

    ms = pymeshlab.MeshSet()
    use_quality = importance is not None
    if use_quality:
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=verts, face_matrix=faces, v_scalar_array=importance,
        ))
    else:
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces))

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target,
        qualitythr=config.quality_threshold,
        preserveboundary=config.preserve_boundary,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        qualityweight=use_quality,
    )
    result = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=result.vertex_matrix(),
        faces=result.face_matrix(),
        process=False,
    )


def _decimate_trimesh(
    mesh: trimesh.Trimesh,
    target: int,
) -> trimesh.Trimesh:
    """Decimate using trimesh's built-in simplification (fast-simplification)."""
    result = mesh.simplify_quadric_decimation(face_count=target)
    return trimesh.Trimesh(
        vertices=result.vertices.copy(),
        faces=result.faces.copy(),
        process=False,
    )


def decimate(
    mesh: trimesh.Trimesh,
    target_faces: int | None = None,
    target_ratio: float | None = None,
    config: DecimateConfig | None = None,
) -> tuple[trimesh.Trimesh, DecimateResult]:
    """QEM decimation with feature preservation.

    pymeshlab if available, trimesh.simplify fallback.
    Does not mutate the input mesh.

    Args:
        mesh: Input triangle mesh.
        target_faces: Absolute target face count.
        target_ratio: Target as fraction of current (0.0-1.0).
        config: Decimation configuration. If None, defaults are used.

    Returns:
        (decimated_mesh, DecimateResult)
    """
    if config is None:
        config = DecimateConfig(
            target_faces=target_faces, target_ratio=target_ratio,
        )
    tf = config.target_faces if target_faces is None else target_faces
    tr = config.target_ratio if target_ratio is None else target_ratio
    target = _resolve_target(mesh, tf, tr)

    n_faces = mesh.faces.shape[0]
    if target >= n_faces:
        # Nothing to do — return a copy
        feat = compute_feature_report(mesh, feature_angle=config.feature_angle)
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

    feat_before = compute_feature_report(mesh, feature_angle=config.feature_angle)
    t0 = time.perf_counter()

    if _HAS_PYMESHLAB:
        backend = Backend.PYMESHLAB
        importance = _vertex_importance(mesh, config.feature_angle)
        result = _decimate_pymeshlab(mesh, target, config, importance=importance)
    else:
        backend = Backend.TRIMESH
        result = _decimate_trimesh(mesh, target)

    duration = (time.perf_counter() - t0) * 1000.0

    feat_after = compute_feature_report(result, feature_angle=config.feature_angle)

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
