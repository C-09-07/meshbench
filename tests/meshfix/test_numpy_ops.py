"""Tests for pure numpy mesh repair operations."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshfix._numpy_ops import (
    remove_degenerates,
    remove_floating_vertices,
    remove_small_shells,
)


class TestRemoveDegenerates:
    def test_removes_degenerate_face(self):
        """Collinear triangle removed, healthy faces kept."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0, 0],  # collinear with v0-v1
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],  # healthy
            [0, 3, 1],  # degenerate (collinear)
        ])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        result = remove_degenerates(mesh)

        assert result.faces.shape[0] == 1
        assert result.vertices.shape[0] == 3

    def test_with_precomputed_indices(self):
        """Pass Defects array directly."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0, 0],
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],  # healthy
            [0, 3, 1],  # degenerate
        ])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        degen_indices = np.array([1], dtype=np.intp)

        result = remove_degenerates(mesh, face_indices=degen_indices)

        assert result.faces.shape[0] == 1
        assert result.vertices.shape[0] == 3

    def test_noop_on_clean_mesh(self, unit_cube):
        """Clean mesh returns identical copy."""
        result = remove_degenerates(unit_cube)

        assert result.faces.shape[0] == unit_cube.faces.shape[0]
        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]
        # Must be a different object
        assert result is not unit_cube

    def test_does_not_mutate_input(self, unit_cube):
        """Input mesh is not modified."""
        orig_faces = unit_cube.faces.copy()
        orig_verts = unit_cube.vertices.copy()

        remove_degenerates(unit_cube)

        np.testing.assert_array_equal(unit_cube.faces, orig_faces)
        np.testing.assert_array_equal(unit_cube.vertices, orig_verts)


class TestRemoveFloatingVertices:
    def test_removes_unreferenced_vertex(self):
        """Unreferenced vertex removed."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [5, 5, 5],  # floating
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        result = remove_floating_vertices(mesh)

        assert result.vertices.shape[0] == 3
        assert result.faces.shape[0] == 1
        # The floating vertex should be gone
        assert not np.any(np.all(np.isclose(result.vertices, [5, 5, 5]), axis=1))

    def test_with_precomputed_indices(self):
        """Pass floating indices directly."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [5, 5, 5],  # floating
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        result = remove_floating_vertices(mesh, vert_indices=np.array([3]))

        assert result.vertices.shape[0] == 3
        assert result.faces.shape[0] == 1

    def test_noop_on_clean_mesh(self, unit_cube):
        """Clean mesh unchanged."""
        result = remove_floating_vertices(unit_cube)

        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]
        assert result.faces.shape[0] == unit_cube.faces.shape[0]


class TestRemoveSmallShells:
    def test_removes_small_shell(self, two_component_mesh):
        """Two-component mesh: add a tiny shell, verify it's removed."""
        # Create a tiny triangle as a small shell
        tiny_verts = np.array([
            [10, 0, 0],
            [10.001, 0, 0],
            [10, 0.001, 0],
        ], dtype=float)
        tiny_faces = np.array([[0, 1, 2]]) + two_component_mesh.vertices.shape[0]

        combined_verts = np.vstack([two_component_mesh.vertices, tiny_verts])
        combined_faces = np.vstack([two_component_mesh.faces, tiny_faces])
        mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)

        result = remove_small_shells(mesh, min_face_count=2, min_face_ratio=0.01)

        # Tiny shell (1 face) should be removed
        assert result.faces.shape[0] < mesh.faces.shape[0]

    def test_keeps_largest_component(self):
        """Always keeps biggest component even below thresholds."""
        a = trimesh.creation.box(extents=[1, 1, 1])
        b = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        b.apply_translation([5, 0, 0])
        combined = trimesh.util.concatenate([a, b])
        mesh = trimesh.Trimesh(
            vertices=combined.vertices,
            faces=combined.faces,
            process=False,
        )

        # Set absurdly high thresholds — largest should still survive
        result = remove_small_shells(mesh, min_face_count=999, min_face_ratio=0.99)

        assert result.faces.shape[0] == a.faces.shape[0]

    def test_single_component_noop(self, unit_cube):
        """Single component mesh untouched."""
        result = remove_small_shells(unit_cube)

        assert result.faces.shape[0] == unit_cube.faces.shape[0]
        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]

    def test_does_not_mutate_input(self, two_component_mesh):
        """Input mesh is not modified."""
        orig_faces = two_component_mesh.faces.copy()

        remove_small_shells(two_component_mesh, min_face_count=999)

        np.testing.assert_array_equal(two_component_mesh.faces, orig_faces)
