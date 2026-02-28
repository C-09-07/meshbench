"""Tests for meshbench._cell (per-face quality metrics)."""

from __future__ import annotations

import numpy as np
import trimesh

import meshbench
from meshbench._cell import (
    compute_cell_report,
    per_face_aspect_ratio,
    per_face_min_angle,
)


class TestAspectRatio:
    def test_equilateral_triangle(self) -> None:
        """Equilateral triangle aspect ratio ≈ 1/sqrt(3)*2 ≈ 1.1547."""
        h = np.sqrt(3) / 2
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        ar = per_face_aspect_ratio(mesh)
        np.testing.assert_allclose(ar[0], 1.0 / np.sqrt(3) * 2, rtol=1e-6)

    def test_degenerate_is_inf(self) -> None:
        """Collinear vertices → zero area → inf aspect ratio."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        ar = per_face_aspect_ratio(mesh)
        assert np.isinf(ar[0])

    def test_cube_uniform(self, unit_cube: trimesh.Trimesh) -> None:
        """All right triangles on a cube should have same aspect ratio."""
        ar = per_face_aspect_ratio(unit_cube)
        assert np.all(np.isfinite(ar))
        np.testing.assert_allclose(ar, ar[0], rtol=1e-6)


class TestMinAngle:
    def test_equilateral_sixty_degrees(self) -> None:
        """All angles of an equilateral triangle are 60°."""
        h = np.sqrt(3) / 2
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        angles = per_face_min_angle(mesh)
        np.testing.assert_allclose(angles[0], 60.0, atol=0.01)

    def test_right_triangle_45(self) -> None:
        """Isosceles right triangle: angles 90°, 45°, 45°. Min = 45°."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        angles = per_face_min_angle(mesh)
        np.testing.assert_allclose(angles[0], 45.0, atol=0.01)

    def test_cube_uniform_angles(self, unit_cube: trimesh.Trimesh) -> None:
        """All right triangles on a cube should have same min angle (45°)."""
        angles = per_face_min_angle(unit_cube)
        np.testing.assert_allclose(angles, 45.0, atol=0.01)


class TestMaxAngle:
    def test_equilateral_sixty(self) -> None:
        """All angles of an equilateral triangle are 60° → max_angle=60."""
        h = np.sqrt(3) / 2
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        np.testing.assert_allclose(report.max_angle_mean, 60.0, atol=0.01)
        np.testing.assert_allclose(report.max_angle_max, 60.0, atol=0.01)

    def test_right_triangle_ninety(self) -> None:
        """Isosceles right triangle: max angle = 90°."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        np.testing.assert_allclose(report.max_angle_max, 90.0, atol=0.01)

    def test_near_degenerate(self) -> None:
        """Very thin triangle → max angle close to 180°."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1e-4, 0]], dtype=float,
        )
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        assert report.max_angle_max > 179.0


class TestSkewness:
    def test_equilateral_zero(self) -> None:
        """Equilateral triangle → skewness = 0."""
        h = np.sqrt(3) / 2
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        np.testing.assert_allclose(report.skewness_mean, 0.0, atol=0.01)
        np.testing.assert_allclose(report.skewness_max, 0.0, atol=0.01)

    def test_right_triangle_half(self) -> None:
        """45-45-90 triangle: skew = max((90-60)/120, (60-45)/60) = max(0.25, 0.25) = 0.25."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        np.testing.assert_allclose(report.skewness_mean, 0.25, atol=0.01)

    def test_near_degenerate_close_to_one(self) -> None:
        """Very thin triangle → skewness close to 1."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1e-4, 0]], dtype=float,
        )
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = compute_cell_report(mesh)
        assert report.skewness_max > 0.99

    def test_skewness_bounded(self, unit_cube: trimesh.Trimesh) -> None:
        """Skewness values are always in [0, 1]."""
        report = compute_cell_report(unit_cube)
        assert 0.0 <= report.skewness_mean <= 1.0
        assert 0.0 <= report.skewness_max <= 1.0


class TestCellReport:
    def test_report_fields(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_cell_report(unit_cube)
        assert report.aspect_ratio_mean > 0
        assert report.aspect_ratio_std >= 0
        assert np.isfinite(report.aspect_ratio_max)
        assert report.min_angle_mean > 0
        assert report.min_angle_min > 0
        assert report.max_angle_mean > 0
        assert report.max_angle_max > 0
        assert 0.0 <= report.skewness_mean <= 1.0

    def test_wired_into_audit(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        assert report.cell is not None
        assert report.cell.aspect_ratio_mean > 0
        assert report.cell.min_angle_mean > 0
        assert report.cell.max_angle_mean > 0
        assert report.cell.skewness_mean >= 0
