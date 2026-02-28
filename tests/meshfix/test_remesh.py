"""Tests for adaptive curvature-weighted remeshing."""

from __future__ import annotations

import numpy as np
import trimesh

from meshfix._remesh import adaptive_remesh


class TestAdaptiveRemesh:
    def test_reduces_face_count(self, icosphere: trimesh.Trimesh):
        n_before = icosphere.faces.shape[0]
        result_mesh, result = adaptive_remesh(icosphere, target_ratio=0.5)
        assert result.faces_after < n_before

    def test_does_not_mutate_input(self, icosphere: trimesh.Trimesh):
        verts_before = icosphere.vertices.copy()
        faces_before = icosphere.faces.copy()
        adaptive_remesh(icosphere, target_ratio=0.5)
        np.testing.assert_array_equal(icosphere.vertices, verts_before)
        np.testing.assert_array_equal(icosphere.faces, faces_before)

    def test_result_has_timing(self, icosphere: trimesh.Trimesh):
        _, result = adaptive_remesh(icosphere, target_ratio=0.5)
        assert result.duration_ms > 0

    def test_falls_back_gracefully(self, icosphere: trimesh.Trimesh):
        """Works regardless of pymeshlab availability."""
        result_mesh, result = adaptive_remesh(icosphere, target_ratio=0.5)
        assert result.faces_after > 0
        assert result.backend is not None

    def test_feature_angle_respected(self, unit_cube: trimesh.Trimesh):
        """Feature edges should survive adaptive remesh on a cube."""
        # Cube has 12 faces, target 10 is mild decimation
        result_mesh, result = adaptive_remesh(
            unit_cube, target_faces=10, feature_angle=45.0,
        )
        assert result.feature_edges_before > 0
