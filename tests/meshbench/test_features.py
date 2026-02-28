"""Tests for feature edge detection via dihedral angles."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench.features import (
    compute_feature_report,
    dihedral_angles,
    feature_edges,
)


class TestDihedralAngles:
    def test_cube_90_degree_edges(self, unit_cube: trimesh.Trimesh):
        """Cube has 12 real edges at ~90° and 6 diagonal edges at ~0°."""
        edges, angles = dihedral_angles(unit_cube)
        assert len(angles) > 0
        # 12 cube edges at 90°, 6 diagonal edges at 0° (coplanar triangles on same quad face)
        n_sharp = np.sum(angles > 45.0)
        n_flat = np.sum(angles < 1.0)
        assert n_sharp == 12
        assert n_flat == 6

    def test_icosphere_all_smooth(self, icosphere: trimesh.Trimesh):
        """Icosphere edges should all be below 30° (smooth surface)."""
        edges, angles = dihedral_angles(icosphere)
        assert len(angles) > 0
        assert np.all(angles < 30.0)

    def test_flat_plane_coplanar(self, flat_plane: trimesh.Trimesh):
        """Interior edges on a flat plane should be ~0°."""
        edges, angles = dihedral_angles(flat_plane)
        # flat_plane has 1 interior edge between the 2 triangles
        assert len(angles) == 1
        np.testing.assert_allclose(angles[0], 0.0, atol=1e-6)

    def test_empty_mesh(self):
        mesh = trimesh.Trimesh()
        edges, angles = dihedral_angles(mesh)
        assert edges.shape == (0, 2)
        assert angles.shape == (0,)


class TestFeatureEdges:
    def test_cube_feature_at_45(self, unit_cube: trimesh.Trimesh):
        """With threshold=45°, cube's 12 real edges are FEATURE (diagonals are smooth)."""
        feat = feature_edges(unit_cube, feature_angle=45.0)
        assert len(feat) == 12  # 12 cube edges at 90°, 6 diagonals at 0° excluded

    def test_cube_smooth_at_91(self, unit_cube: trimesh.Trimesh):
        """With threshold=91°, cube edges (90°) are SMOOTH."""
        feat = feature_edges(unit_cube, feature_angle=91.0)
        assert len(feat) == 0


class TestFeatureReport:
    def test_open_box_has_boundary(self, open_box: trimesh.Trimesh):
        """Open box should have boundary edges counted."""
        report = compute_feature_report(open_box)
        assert report.boundary_edge_count > 0

    def test_feature_report_serializable(self, unit_cube: trimesh.Trimesh):
        report = compute_feature_report(unit_cube, include_angles=True)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "feature_edge_count" in d
        assert "dihedral_angles" in d
        assert isinstance(d["dihedral_angles"], list)

    def test_empty_mesh(self):
        mesh = trimesh.Trimesh()
        report = compute_feature_report(mesh)
        assert report.feature_edge_count == 0
        assert report.smooth_edge_count == 0
        assert report.boundary_edge_count == 0
        assert report.feature_edge_ratio == 0.0
