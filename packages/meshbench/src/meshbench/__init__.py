"""meshbench — Geometric mesh quality analysis library.

Public API:
    audit(mesh)       → MeshReport (full analysis)
    fingerprint(mesh) → Fingerprint
    manifold(mesh)    → ManifoldReport
    load(path)        → trimesh.Trimesh
"""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._cell import CellArrays, compute_cell_arrays, compute_cell_report
from meshbench._edges import (
    EdgeData,
    EdgeFaceMap,
    build_edge_face_map,
    edge_data_from_map,
    edge_face_counts,
)
from meshbench._intersections import detect_self_intersections
from meshbench.density import compute_density_report
from meshbench.fingerprint import compute_fingerprint, compute_pca
from meshbench.loading import load
from meshbench.manifold import (
    boundary_edges,
    compute_manifold_report,
    non_manifold_edges,
    non_manifold_vertices,
)
from meshbench.normals import compute_normal_report, flipped_face_pairs
from meshbench.scoring import compute_score
from meshbench.topology import (
    compute_topology_report,
    degenerate_faces,
    floating_vertices,
    high_valence_vertices,
)
from meshbench.types import (
    CellReport,
    Check,
    Defects,
    DensityReport,
    Fingerprint,
    ManifoldReport,
    MeshReport,
    NormalReport,
    ScoreCard,
    TopologyReport,
)


def _build_defects(
    mesh: trimesh.Trimesh,
    cell_arrays: CellArrays,
    edge_data: EdgeData,
    efm: EdgeFaceMap | None,
) -> Defects:
    """Collect per-element defect indices and per-face metric arrays."""
    ar, min_angles, max_angles, skew = cell_arrays

    b_edges = boundary_edges(mesh, edge_data)
    nm_edges = non_manifold_edges(mesh, edge_data)
    nm_verts = non_manifold_vertices(mesh, _edge_face_map=efm)
    _, si_pairs = detect_self_intersections(mesh)
    degen = degenerate_faces(mesh)
    floating = floating_vertices(mesh)
    flipped = flipped_face_pairs(mesh)
    hv_verts = high_valence_vertices(mesh, _edge_data=edge_data)

    return Defects(
        boundary_edges=b_edges if len(b_edges) > 0 else None,
        non_manifold_edges=nm_edges if len(nm_edges) > 0 else None,
        non_manifold_vertices=np.array(nm_verts, dtype=np.intp) if nm_verts else None,
        self_intersection_pairs=si_pairs if si_pairs.shape[0] > 0 else None,
        flipped_face_pairs=flipped if len(flipped) > 0 else None,
        degenerate_faces=np.array(degen, dtype=np.intp) if degen else None,
        floating_vertices=np.array(floating, dtype=np.intp) if floating else None,
        high_valence_vertices=hv_verts if len(hv_verts) > 0 else None,
        aspect_ratio=ar,
        min_angle=min_angles,
        max_angle=max_angles,
        skewness=skew,
    )


def audit(
    mesh: trimesh.Trimesh,
    include_defects: bool = False,
) -> MeshReport:
    """Run full mesh quality analysis.

    Args:
        mesh: Input triangle mesh.
        include_defects: If True, populate MeshReport.defects with
            per-element defect indices and per-face metric arrays.

    Returns a MeshReport with fingerprint, manifold, normals,
    topology, density, cell, and optionally defects sub-reports.
    """
    # Compute shared data once — build rich edge map, derive EdgeData from it
    n_faces = mesh.faces.shape[0]
    if n_faces > 0:
        efm = build_edge_face_map(mesh.faces)
        edges, counts = edge_data_from_map(efm)
    else:
        efm = None
        edges, counts = edge_face_counts(mesh)
    edge_data = (edges, counts)
    pca = compute_pca(mesh)

    # Compute topology first to share component_count with fingerprint
    topo = compute_topology_report(mesh, _edge_data=edge_data)

    # Compute cell arrays once, share with both cell report and defects
    cell_arrays = compute_cell_arrays(mesh)
    cell = compute_cell_report(mesh, _cell_arrays=cell_arrays)

    defects = None
    if include_defects:
        defects = _build_defects(mesh, cell_arrays, edge_data, efm)

    return MeshReport(
        vertex_count=mesh.vertices.shape[0],
        face_count=n_faces,
        edge_count=len(edges),
        fingerprint=compute_fingerprint(
            mesh, _pca=pca, _component_count=topo.component_count,
        ),
        manifold=compute_manifold_report(
            mesh, _edge_data=edge_data, _edge_face_map=efm,
        ),
        normals=compute_normal_report(mesh, _pca=pca),
        topology=topo,
        density=compute_density_report(mesh),
        cell=cell,
        defects=defects,
    )


def fingerprint(mesh: trimesh.Trimesh) -> Fingerprint:
    """Compute shape fingerprint only."""
    return compute_fingerprint(mesh)


def manifold(mesh: trimesh.Trimesh) -> ManifoldReport:
    """Compute manifold analysis only."""
    return compute_manifold_report(mesh)


def score(report: MeshReport, profile: str = "print") -> ScoreCard:
    """Score a MeshReport against an application profile.

    Args:
        report: Full mesh analysis report from audit().
        profile: One of "print", "animation", "simulation", "ai_output".

    Returns:
        ScoreCard with grade, score, and individual check results.
    """
    return compute_score(report, profile)


__all__ = [
    "audit",
    "fingerprint",
    "load",
    "manifold",
    "score",
    "CellReport",
    "Check",
    "Defects",
    "DensityReport",
    "Fingerprint",
    "ManifoldReport",
    "MeshReport",
    "NormalReport",
    "ScoreCard",
    "TopologyReport",
]
