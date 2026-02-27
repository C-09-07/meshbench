"""Tests for meshbench.topology."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench.topology import (
    component_sizes,
    compute_topology_report,
    degenerate_faces,
    floating_vertices,
    genus,
    valence_histogram,
    valence_stats,
)


class TestGenus:
    def test_sphere_genus_zero(self, icosphere: trimesh.Trimesh) -> None:
        g = genus(icosphere)
        assert g == 0

    def test_cube_genus_zero(self, unit_cube: trimesh.Trimesh) -> None:
        g = genus(unit_cube)
        assert g == 0


class TestValence:
    def test_histogram_not_empty(self, unit_cube: trimesh.Trimesh) -> None:
        hist = valence_histogram(unit_cube)
        assert len(hist) > 0
        assert sum(hist.values()) == unit_cube.vertices.shape[0]

    def test_stats_reasonable(self, unit_cube: trimesh.Trimesh) -> None:
        mean, std, quad_ratio, outliers = valence_stats(unit_cube)
        assert mean > 0
        assert std >= 0
        assert 0 <= quad_ratio <= 1
        assert outliers >= 0

    def test_icosphere_valence(self, icosphere: trimesh.Trimesh) -> None:
        hist = valence_histogram(icosphere)
        # Icosphere vertices have valence 5 or 6
        for val in hist:
            assert val >= 3


class TestDegenerateFaces:
    def test_cube_no_degenerates(self, unit_cube: trimesh.Trimesh) -> None:
        degen = degenerate_faces(unit_cube)
        assert len(degen) == 0

    def test_degenerate_triangle(self) -> None:
        """A zero-area triangle (collinear vertices) should be detected."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],  # collinear
            [0, 1, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        degen = degenerate_faces(mesh)
        assert 0 in degen  # first face is degenerate
        assert 1 not in degen


class TestFloatingVertices:
    def test_cube_no_floating(self, unit_cube: trimesh.Trimesh) -> None:
        floating = floating_vertices(unit_cube)
        assert len(floating) == 0

    def test_added_floating_vertex(self, unit_cube: trimesh.Trimesh) -> None:
        """Add a vertex not referenced by any face."""
        new_verts = np.vstack([unit_cube.vertices, [10, 10, 10]])
        mesh = trimesh.Trimesh(vertices=new_verts, faces=unit_cube.faces, process=False)
        floating = floating_vertices(mesh)
        assert len(floating) == 1
        assert floating[0] == len(unit_cube.vertices)


class TestComponentSizes:
    def test_single_component(self, unit_cube: trimesh.Trimesh) -> None:
        sizes = component_sizes(unit_cube)
        assert len(sizes) == 1

    def test_two_components(self, two_component_mesh: trimesh.Trimesh) -> None:
        sizes = component_sizes(two_component_mesh)
        assert len(sizes) == 2
        # Both should have same size (two identical cubes)
        assert sizes[0] == sizes[1]


class TestTopologyReport:
    def test_report_fields(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_topology_report(unit_cube)
        assert report.genus >= 0
        assert report.valence_mean > 0
        assert report.valence_std >= 0
        assert len(report.valence_histogram) > 0
        assert report.degenerate_face_count == 0
        assert report.floating_vertex_count == 0
        assert report.component_count == 1

    def test_two_components_report(self, two_component_mesh: trimesh.Trimesh) -> None:
        report = compute_topology_report(two_component_mesh)
        assert report.component_count == 2

    def test_clean_mesh_ratios_zero(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_topology_report(unit_cube)
        assert report.degenerate_face_ratio == 0.0
        assert report.floating_vertex_ratio == 0.0

    def test_degenerate_ratio(self) -> None:
        """Mesh with 1 degenerate + 1 good face → ratio = 0.5."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],  # collinear with first two
            [0, 1, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_topology_report(mesh)
        assert report.degenerate_face_count == 1
        assert report.degenerate_face_ratio == pytest.approx(0.5)
