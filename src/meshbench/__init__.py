"""meshbench — Geometric mesh quality analysis library.

Public API:
    audit(mesh)       → MeshReport (full analysis)
    fingerprint(mesh) → Fingerprint
    manifold(mesh)    → ManifoldReport
    compare(a, b)     → NotImplementedError (placeholder)
"""

from __future__ import annotations

import trimesh

from meshbench._edges import edge_face_counts
from meshbench.density import compute_density_report
from meshbench.fingerprint import compute_fingerprint
from meshbench.manifold import compute_manifold_report
from meshbench.normals import compute_normal_report
from meshbench.topology import compute_topology_report
from meshbench.types import (
    DensityReport,
    Fingerprint,
    ManifoldReport,
    MeshReport,
    NormalReport,
    TopologyReport,
)


def audit(mesh: trimesh.Trimesh) -> MeshReport:
    """Run full mesh quality analysis.

    Returns a MeshReport with fingerprint, manifold, normals,
    topology, and density sub-reports.
    """
    # Compute edges once, share across sub-reports
    edges, counts = edge_face_counts(mesh)

    return MeshReport(
        vertex_count=mesh.vertices.shape[0],
        face_count=mesh.faces.shape[0],
        edge_count=len(edges),
        fingerprint=compute_fingerprint(mesh),
        manifold=compute_manifold_report(mesh, _edge_data=(edges, counts)),
        normals=compute_normal_report(mesh),
        topology=compute_topology_report(mesh, _edge_data=(edges, counts)),
        density=compute_density_report(mesh),
    )


def fingerprint(mesh: trimesh.Trimesh) -> Fingerprint:
    """Compute shape fingerprint only."""
    return compute_fingerprint(mesh)


def manifold(mesh: trimesh.Trimesh) -> ManifoldReport:
    """Compute manifold analysis only."""
    return compute_manifold_report(mesh)


def compare(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> None:
    """Compare two meshes (placeholder — not yet implemented)."""
    raise NotImplementedError(
        "meshbench.compare() is planned but not yet implemented."
    )


__all__ = [
    "audit",
    "fingerprint",
    "manifold",
    "compare",
    "MeshReport",
    "Fingerprint",
    "ManifoldReport",
    "NormalReport",
    "TopologyReport",
    "DensityReport",
]
