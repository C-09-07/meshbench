"""Tests for meshbench.density."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.density import (
    compute_density_report,
    face_density_variance,
    vertices_per_area,
)


class TestVerticesPerArea:
    def test_positive(self, unit_cube: trimesh.Trimesh) -> None:
        vpa = vertices_per_area(unit_cube)
        assert vpa > 0

    def test_icosphere_higher_density(
        self, icosphere: trimesh.Trimesh, unit_cube: trimesh.Trimesh
    ) -> None:
        # Icosphere has many more vertices per unit area than a cube
        ico_vpa = vertices_per_area(icosphere)
        cube_vpa = vertices_per_area(unit_cube)
        assert ico_vpa > cube_vpa


class TestFaceDensityVariance:
    def test_cube_low_variance(self, unit_cube: trimesh.Trimesh) -> None:
        # All cube faces should have similar area
        var = face_density_variance(unit_cube)
        assert var < 0.01

    def test_mixed_faces(self) -> None:
        """Mesh with very different face sizes should have high variance."""
        verts = np.array([
            [0, 0, 0],
            [0.001, 0, 0],
            [0, 0.001, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        var = face_density_variance(mesh)
        assert var > 0.9  # High variance due to extreme size difference

    def test_single_face(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        assert face_density_variance(mesh) == 0.0


class TestDensityReport:
    def test_report_fields(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_density_report(unit_cube)
        assert report.vertices_per_area > 0
        assert report.face_density_variance >= 0
