"""meshbench — Geometric mesh quality analysis library.

Public API:
    audit(mesh)       → MeshReport (full analysis)
    fingerprint(mesh) → Fingerprint
    manifold(mesh)    → ManifoldReport
    load(path)        → trimesh.Trimesh
"""

from __future__ import annotations

import trimesh

from meshbench._cell import compute_cell_report
from meshbench._edges import build_edge_face_map, edge_data_from_map, edge_face_counts
from meshbench.density import compute_density_report
from meshbench.fingerprint import compute_fingerprint, compute_pca
from meshbench.loading import load
from meshbench.manifold import compute_manifold_report
from meshbench.normals import compute_normal_report
from meshbench.scoring import compute_score
from meshbench.topology import compute_topology_report
from meshbench.types import (
    CellReport,
    Check,
    DensityReport,
    Fingerprint,
    ManifoldReport,
    MeshReport,
    NormalReport,
    ScoreCard,
    TopologyReport,
)


def audit(mesh: trimesh.Trimesh) -> MeshReport:
    """Run full mesh quality analysis.

    Returns a MeshReport with fingerprint, manifold, normals,
    topology, density, and cell sub-reports.
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
        cell=compute_cell_report(mesh),
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
    "DensityReport",
    "Fingerprint",
    "ManifoldReport",
    "MeshReport",
    "NormalReport",
    "ScoreCard",
    "TopologyReport",
]
