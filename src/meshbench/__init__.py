"""meshbench — Geometric mesh quality analysis library.

Public API:
    audit(mesh)       → MeshReport (full analysis)
    fingerprint(mesh) → Fingerprint
    manifold(mesh)    → ManifoldReport
    compare(a, b)     → NotImplementedError (placeholder)
"""

from __future__ import annotations

import trimesh

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
    # Count unique edges
    edge_set: set[tuple[int, int]] = set()
    for face in mesh.faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_set.add((min(v0, v1), max(v0, v1)))

    return MeshReport(
        vertex_count=mesh.vertices.shape[0],
        face_count=mesh.faces.shape[0],
        edge_count=len(edge_set),
        fingerprint=compute_fingerprint(mesh),
        manifold=compute_manifold_report(mesh),
        normals=compute_normal_report(mesh),
        topology=compute_topology_report(mesh),
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
