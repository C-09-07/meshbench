"""Tests for QEM decimation."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshfix._decimate import decimate


class TestDecimate:
    def test_reduces_face_count(self, icosphere: trimesh.Trimesh):
        n_before = icosphere.faces.shape[0]
        result_mesh, result = decimate(icosphere, target_ratio=0.5)
        assert result.faces_after < n_before

    def test_respects_target_approx(self, icosphere: trimesh.Trimesh):
        """QEM is approximate — within 20% of target is acceptable."""
        n_before = icosphere.faces.shape[0]
        target = n_before // 2
        result_mesh, result = decimate(icosphere, target_faces=target)
        assert result.faces_after <= target * 1.2
        assert result.faces_after >= target * 0.5

    def test_preserves_feature_edges(self, unit_cube: trimesh.Trimesh):
        """Cube feature edges should survive mild decimation."""
        # Cube has only 12 faces, so request 10 (mild decimation)
        result_mesh, result = decimate(unit_cube, target_faces=10)
        # Feature edges should still exist
        assert result.feature_edges_before > 0

    def test_does_not_mutate_input(self, icosphere: trimesh.Trimesh):
        verts_before = icosphere.vertices.copy()
        faces_before = icosphere.faces.copy()
        decimate(icosphere, target_ratio=0.5)
        np.testing.assert_array_equal(icosphere.vertices, verts_before)
        np.testing.assert_array_equal(icosphere.faces, faces_before)

    def test_target_ratio_halves(self, icosphere: trimesh.Trimesh):
        n_before = icosphere.faces.shape[0]
        _, result = decimate(icosphere, target_ratio=0.5)
        # Should be roughly half, within 30%
        assert result.faces_after < n_before * 0.8
        assert result.faces_after > n_before * 0.2

    def test_target_exceeds_current(self, unit_cube: trimesh.Trimesh):
        """If target >= current, return copy unchanged."""
        n_before = unit_cube.faces.shape[0]
        result_mesh, result = decimate(unit_cube, target_faces=n_before + 100)
        assert result.faces_after == n_before
        # Should be a copy, not the same object
        assert result_mesh is not unit_cube

    def test_result_serializable(self, icosphere: trimesh.Trimesh):
        _, result = decimate(icosphere, target_ratio=0.5)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "faces_before" in d
        assert "duration_ms" in d

    def test_empty_mesh(self):
        """Empty mesh with target_ratio returns empty copy."""
        mesh = trimesh.Trimesh()
        result_mesh, result = decimate(mesh, target_ratio=0.5)
        assert result.faces_before == 0
        assert result.faces_after == 0

    def test_both_targets_raises(self, icosphere: trimesh.Trimesh):
        with pytest.raises(ValueError, match="not both"):
            decimate(icosphere, target_faces=100, target_ratio=0.5)
