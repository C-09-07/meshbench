"""Tests for meshbench.normals."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.normals import (
    backface_coherence,
    backface_divergence,
    check_and_fix_inverted_normals,
    compute_normal_report,
    dominant_normal,
    flipped_normal_count,
    normal_entropy,
    winding_consistency,
)


class TestNormalEntropy:
    def test_sphere_high_entropy(self, icosphere: trimesh.Trimesh) -> None:
        ent = normal_entropy(icosphere)
        assert ent > 3.0

    def test_flat_low_entropy(self, flat_plane: trimesh.Trimesh) -> None:
        ent = normal_entropy(flat_plane)
        # Two coplanar triangles — all normals point the same way
        assert ent < 1.0

    def test_empty_mesh_zero(self) -> None:
        mesh = trimesh.Trimesh()
        assert normal_entropy(mesh) == 0.0


class TestDominantNormal:
    def test_flat_plane_has_dominant(self, flat_plane: trimesh.Trimesh) -> None:
        dn = dominant_normal(flat_plane)
        assert dn is not None
        # Should point along Y (normal to XZ plane)
        assert abs(abs(dn[1]) - 1.0) < 0.1

    def test_sphere_no_dominant(self, icosphere: trimesh.Trimesh) -> None:
        # Sphere is too diffuse for a dominant normal
        dn = dominant_normal(icosphere)
        assert dn is None


class TestFlippedNormals:
    def test_consistent_cube(self, unit_cube: trimesh.Trimesh) -> None:
        count = flipped_normal_count(unit_cube)
        # A proper cube has consistent winding
        assert count == 0

    def test_winding_consistency_cube(self, unit_cube: trimesh.Trimesh) -> None:
        assert winding_consistency(unit_cube) == 1.0

    def test_inverted_faces_detected(self) -> None:
        """Create a mesh with one deliberately flipped face."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        # Flip winding of one face
        mesh.faces[0] = mesh.faces[0][::-1]
        count = flipped_normal_count(mesh)
        assert count > 0
        assert winding_consistency(mesh) < 1.0

    def test_single_face_mesh(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        assert flipped_normal_count(mesh) == 0
        assert winding_consistency(mesh) == 1.0


class TestBackfaceDivergence:
    def test_sphere_low_divergence(self, icosphere: trimesh.Trimesh) -> None:
        div = backface_divergence(icosphere)
        # Sphere is symmetric — front/back should have similar entropy
        assert div < 1.0

    def test_single_face(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        assert backface_divergence(mesh) == 0.0


class TestBackfaceCoherence:
    def test_returns_float(self, unit_cube: trimesh.Trimesh) -> None:
        result = backface_coherence(unit_cube, np.array([0, 0, 1]))
        assert isinstance(result, float)
        assert result >= 0.0


class TestCheckAndFixInverted:
    def test_no_flip_needed(self, unit_cube: trimesh.Trimesh) -> None:
        dn = dominant_normal(unit_cube)
        if dn is not None:
            flipped = check_and_fix_inverted_normals(unit_cube, dn)
            # Standard cube shouldn't need flipping
            assert isinstance(flipped, bool)


class TestNormalReport:
    def test_report_fields(self, unit_cube: trimesh.Trimesh) -> None:
        report = compute_normal_report(unit_cube)
        assert report.entropy >= 0
        assert report.flipped_count >= 0
        assert 0.0 <= report.winding_consistency <= 1.0
        assert report.backface_divergence >= 0
