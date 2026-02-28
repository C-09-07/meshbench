"""Tests for meshbench.manifold."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.manifold import (
    boundary_edges,
    boundary_loops,
    compute_manifold_report,
    non_manifold_edges,
    non_manifold_vertices,
)


class TestBoundaryEdges:
    def test_closed_cube_no_boundary(self, unit_cube: trimesh.Trimesh) -> None:
        edges = boundary_edges(unit_cube)
        assert len(edges) == 0

    def test_open_box_has_boundary(self, open_box: trimesh.Trimesh) -> None:
        edges = boundary_edges(open_box)
        assert len(edges) > 0

    def test_flat_plane_has_boundary(self, flat_plane: trimesh.Trimesh) -> None:
        edges = boundary_edges(flat_plane)
        # A flat quad has 4 boundary edges (outer rim) minus 0 internal shared
        # Actually: 4 outer edges + 1 internal shared. The internal diagonal is shared by 2 faces.
        # So boundary = 4 edges shared by exactly 1 face.
        assert len(edges) == 4


class TestBoundaryLoops:
    def test_closed_cube_zero_loops(self, unit_cube: trimesh.Trimesh) -> None:
        assert boundary_loops(unit_cube) == 0

    def test_open_box_one_loop(self, open_box: trimesh.Trimesh) -> None:
        loops = boundary_loops(open_box)
        # Removing one face from a cube creates one hole → one loop
        assert loops == 1

    def test_flat_plane_one_loop(self, flat_plane: trimesh.Trimesh) -> None:
        # The flat quad's boundary edges form one connected loop
        assert boundary_loops(flat_plane) == 1


class TestNonManifoldEdges:
    def test_cube_no_nonmanifold(self, unit_cube: trimesh.Trimesh) -> None:
        edges = non_manifold_edges(unit_cube)
        assert len(edges) == 0

    def test_bowtie_edge(self) -> None:
        """Three triangles sharing one edge → non-manifold."""
        verts = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0.5, 1, 0],  # 2
            [0.5, -1, 0],  # 3
            [0.5, 0, 1],  # 4
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        nm = non_manifold_edges(mesh)
        # Edge (0, 1) is shared by 3 faces
        assert len(nm) >= 1
        assert (0, 1) in nm


class TestNonManifoldVertices:
    def test_cube_no_nonmanifold_verts(self, unit_cube: trimesh.Trimesh) -> None:
        verts = non_manifold_vertices(unit_cube)
        assert len(verts) == 0


class TestManifoldReport:
    def test_closed_cube(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_manifold_report(unit_cube)
        assert report.is_watertight is True
        assert report.boundary_edge_count == 0
        assert report.boundary_loop_count == 0
        assert report.non_manifold_edge_count == 0

    def test_open_box(self, open_box: trimesh.Trimesh) -> None:
        report = compute_manifold_report(open_box)
        assert report.is_watertight is False
        assert report.boundary_edge_count > 0
        assert report.boundary_loop_count >= 1

    def test_sphere_watertight(self, icosphere: trimesh.Trimesh) -> None:
        report = compute_manifold_report(icosphere)
        assert report.is_watertight is True

    def test_closed_cube_ratios_zero(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_manifold_report(unit_cube)
        assert report.non_manifold_edge_ratio == 0.0
        assert report.non_manifold_vertex_ratio == 0.0
        assert report.boundary_edge_ratio == 0.0

    def test_open_box_boundary_ratio(self, open_box: trimesh.Trimesh) -> None:
        report = compute_manifold_report(open_box)
        assert report.boundary_edge_ratio > 0.0
        assert report.boundary_edge_ratio <= 1.0
