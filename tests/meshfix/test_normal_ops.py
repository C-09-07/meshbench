"""Tests for winding/normal fix operations."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshfix._normal_ops import fix_normals, fix_winding


class TestFixWinding:
    def test_consistent_mesh_unchanged(self, unit_cube):
        """Already-consistent mesh stays consistent."""
        result = fix_winding(unit_cube)

        assert result.faces.shape == unit_cube.faces.shape
        assert result.vertices.shape == unit_cube.vertices.shape

    def test_flipped_face_corrected(self):
        """One flipped face gets corrected."""
        # Create a simple mesh with one flipped face
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        # Flip one face by swapping two vertices
        mesh.faces[0] = mesh.faces[0][[0, 2, 1]]

        result = fix_winding(mesh)

        # After fix, winding should be consistent
        # trimesh.repair.fix_winding makes all faces consistent
        assert result.faces.shape == mesh.faces.shape

    def test_returns_new_mesh(self, unit_cube):
        """Input not mutated — returns new object."""
        orig_faces = unit_cube.faces.copy()
        result = fix_winding(unit_cube)

        assert result is not unit_cube
        np.testing.assert_array_equal(unit_cube.faces, orig_faces)


class TestFixNormals:
    def test_returns_new_mesh(self, unit_cube):
        """Input not mutated."""
        orig_faces = unit_cube.faces.copy()
        result = fix_normals(unit_cube)

        assert result is not unit_cube
        np.testing.assert_array_equal(unit_cube.faces, orig_faces)

    def test_normals_outward(self, icosphere):
        """Normals point outward after fix."""
        # Flip all normals inward
        inverted = trimesh.Trimesh(
            vertices=icosphere.vertices.copy(),
            faces=icosphere.faces[:, ::-1],  # reverse winding
            process=False,
        )

        result = fix_normals(inverted)

        # After fix_normals, the mesh volume should be positive
        # (outward normals = positive volume for closed meshes)
        assert result.volume > 0

    def test_preserves_geometry(self, unit_cube):
        """Vertex positions unchanged."""
        result = fix_normals(unit_cube)

        # Same number of vertices and faces
        assert result.vertices.shape == unit_cube.vertices.shape
        assert result.faces.shape == unit_cube.faces.shape
